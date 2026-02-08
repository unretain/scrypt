// Copyright (c) 2012-2013 The PPCoin developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
#ifdef _MSC_VER
    #include <stdint.h>

    #include "msvc_warnings.push.h"
#endif

#include "kernel.h"

#include "chainparams.h"
#include "validation.h"
#include "txdb.h"
#include "consensus/validation.h"
#include "pow.h"
#include "streams.h"
#include "policy/policy.h"
#include "script/interpreter.h"

#include <boost/assign/list_of.hpp>

using std::min;
using std::vector;
using std::pair;
using std::map;
using std::make_pair;
using std::string;

typedef std::map<int, unsigned int> MapModifierCheckpoints;

// Hard checkpoints of stake modifiers to ensure they are deterministic
// Empty for new Scrypt chain - will be populated after mainnet launch
static std::map<int, unsigned int> mapStakeModifierCheckpoints;

// Hard checkpoints of stake modifiers to ensure they are deterministic (testNet)
static std::map<int, unsigned int> mapStakeModifierCheckpointsTestNet;

// Whether the given block is subject to new modifier protocol
bool IsFixedModifierInterval(unsigned int nTimeBlock)
{
    return (nTimeBlock >= (fTestNet? nModifierTestSwitchTime : nModifierSwitchTime));
}

// Get time weight
::int64_t GetWeight(int64_t nIntervalBeginning, ::int64_t nIntervalEnd)
{
    const Consensus::Params& params = Params().GetConsensus();
    // Kernel hash weight starts from 0 at the 30-day min age
    // this change increases active coins participating the hash and helps
    // to secure the network when proof-of-stake difficulty is low
    //
    // Maximum TimeWeight is 90 days.

    return min(nIntervalEnd - nIntervalBeginning - params.nStakeMinAge, (int64_t)params.nStakeMaxAge);
}

// Get the last stake modifier and its generation time from a given block
static bool GetLastStakeModifier(const CBlockIndex* pindex, uint64_t& nStakeModifier, ::int64_t& nModifierTime)
{
    if (!pindex)
        return error("GetLastStakeModifier: null pindex");
    while (pindex && pindex->pprev && !pindex->GeneratedStakeModifier())
        pindex = pindex->pprev;
    if (!pindex->GeneratedStakeModifier())
        return error("GetLastStakeModifier: no generation at genesis block");
    nStakeModifier = pindex->nStakeModifier;
    nModifierTime = pindex->GetBlockTime();
    return true;
}

// Get selection interval section (in seconds)
static ::int64_t GetStakeModifierSelectionIntervalSection(int nSection)
{
    Yassert (nSection >= 0 && nSection < 64);
    return (Params().GetConsensus().nModifierInterval * 63 / (63 + ((63 - nSection) * (MODIFIER_INTERVAL_RATIO - 1))));
}

// Get stake modifier selection interval (in seconds)
static ::int64_t GetStakeModifierSelectionInterval()
{
    ::int64_t 
        nSelectionInterval = 0;
    for (int nSection = 0; nSection < 64; ++nSection)
        nSelectionInterval += GetStakeModifierSelectionIntervalSection(nSection);
    return nSelectionInterval;
}

// select a block from the candidate blocks in vSortedByTimestamp, excluding
// already selected blocks in vSelectedBlocks, and with timestamp up to
// nSelectionIntervalStop.
static bool SelectBlockFromCandidates(
                                      vector< pair< ::int64_t, uint256 > >& vSortedByTimestamp,
                                      map<uint256, const CBlockIndex* >& mapSelectedBlocks,
                                      ::int64_t nSelectionIntervalStop,
                                      ::uint64_t nStakeModifierPrev,
                                      const CBlockIndex** pindexSelected
                                     )
{
    bool fSelected = false;
    uint256 hashBest = 0;
    *pindexSelected = (const CBlockIndex*) 0;
    for(const PAIRTYPE(int64_t, uint256)& item : vSortedByTimestamp)
    {
        if (!mapBlockIndex.count(item.second))
            return error("SelectBlockFromCandidates: failed to find block index for candidate block %s", item.second.ToString().c_str());
        const CBlockIndex* pindex = mapBlockIndex[item.second];
        if (fSelected && pindex->GetBlockTime() > nSelectionIntervalStop)
            break;
        if (mapSelectedBlocks.count(pindex->GetBlockHash()) > 0)
            continue;
        // compute the selection hash by hashing its proof-hash and the
        // previous proof-of-stake modifier
        uint256 hashProof = pindex->IsProofOfStake()? pindex->hashProofOfStake : pindex->GetBlockHash();
        CDataStream ss(SER_GETHASH, 0);
        ss << hashProof << nStakeModifierPrev;
        uint256 hashSelection = Hash(ss.begin(), ss.end());
        // the selection hash is divided by 2**32 so that proof-of-stake block
        // is always favored over proof-of-work block. this is to preserve
        // the energy efficiency property
        if (pindex->IsProofOfStake())
            hashSelection >>= 32;
        if (fSelected && hashSelection < hashBest)
        {
            hashBest = hashSelection;
            *pindexSelected = (const CBlockIndex*) pindex;
        }
        else if (!fSelected)
        {
            fSelected = true;
            hashBest = hashSelection;
            *pindexSelected = (const CBlockIndex*) pindex;
        }
    }
    if (fDebug && gArgs.GetBoolArg("-printstakemodifier"))
        LogPrintf("SelectBlockFromCandidates: selection hash=%s\n", hashBest.ToString());
    return fSelected;
}

// Stake Modifier (hash modifier of proof-of-stake):
// The purpose of stake modifier is to prevent a txout (coin) owner from
// computing future proof-of-stake generated by this txout at the time
// of transaction confirmation. To meet kernel protocol, the txout
// must hash with a future stake modifier to generate the proof.
// Stake modifier consists of bits each of which is contributed from a
// selected block of a given block group in the past.
// The selection of a block is based on a hash of the block's proof-hash and
// the previous stake modifier.
// Stake modifier is recomputed at a fixed time interval instead of every 
// block. This is to make it difficult for an attacker to gain control of
// additional bits in the stake modifier, even after generating a chain of
// blocks.
bool ComputeNextStakeModifier(const CBlockIndex* pindexCurrent, uint64_t& nStakeModifier, bool& fGeneratedStakeModifier)
{
    if (chainActive.Tip() && (chainActive.Tip()->nHeight + 1) >= nMainnetNewLogicBlockNumber)
    {
        return true;
    }

    const Consensus::Params& params = Params().GetConsensus();
    nStakeModifier = 0;
    fGeneratedStakeModifier = false;
    const CBlockIndex* pindexPrev = pindexCurrent->pprev;
    if (!pindexPrev)
    {
        fGeneratedStakeModifier = true;
        return true;  // genesis block's modifier is 0
    }

    // First find current stake modifier and its generation block time
    // if it's not old enough, return the same stake modifier
    ::int64_t nModifierTime = 0;
    if (!GetLastStakeModifier(pindexPrev, nStakeModifier, nModifierTime))
        return error("ComputeNextStakeModifier: unable to get last modifier");
    if (fDebug)
    {
      LogPrintf("ComputeNextStakeModifier: prev modifier=0x%016" PRIx64
                " time=%s epoch=%u\n",
                nStakeModifier, DateTimeStrFormat(nModifierTime),
                (unsigned int)nModifierTime);
    }
    if (nModifierTime / params.nModifierInterval >= pindexPrev->GetBlockTime() / params.nModifierInterval)
    {
        if (fDebug)
        {
          LogPrintf(
              "ComputeNextStakeModifier: no new interval keep current "
              "modifier: pindexPrev nHeight=%d nTime=%u\n",
              pindexPrev->nHeight, (unsigned int)pindexPrev->GetBlockTime());
        }
        return true;
    }
    if (nModifierTime / params.nModifierInterval >= pindexCurrent->GetBlockTime() / params.nModifierInterval)
    {
        // fixed interval protocol requires current block timestamp also be in a different modifier interval
        if (IsFixedModifierInterval(pindexCurrent->nTime))
        {
            if (fDebug)
            {
              LogPrintf(
                  "ComputeNextStakeModifier: no new interval keep current "
                  "modifier: pindexCurrent nHeight=%d nTime=%u\n",
                  pindexCurrent->nHeight,
                  (unsigned int)pindexCurrent->GetBlockTime());
            }
            return true;
        }
        else
        {
            if (fDebug)
            {
              LogPrintf(
                  "ComputeNextStakeModifier: old modifier at block %s not "
                  "meeting fixed modifier interval: pindexCurrent nHeight=%d "
                  "nTime=%u\n",
                  pindexCurrent->GetBlockHash().ToString(),
                  pindexCurrent->nHeight,
                  (unsigned int)pindexCurrent->GetBlockTime());
            }
        }
    }

    // Sort candidate blocks by timestamp
    vector<pair< ::int64_t, uint256> > vSortedByTimestamp;
    vSortedByTimestamp.reserve(64 * params.nModifierInterval / nStakeTargetSpacing);
    ::int64_t nSelectionInterval = GetStakeModifierSelectionInterval();
    ::int64_t nSelectionIntervalStart = (pindexPrev->GetBlockTime() / params.nModifierInterval) * params.nModifierInterval - nSelectionInterval;
    const CBlockIndex* pindex = pindexPrev;
    while (pindex && pindex->GetBlockTime() >= nSelectionIntervalStart)
    {
        vSortedByTimestamp.push_back(make_pair(pindex->GetBlockTime(), pindex->GetBlockHash()));
        pindex = pindex->pprev;
    }
    int nHeightFirstCandidate = pindex ? (pindex->nHeight + 1) : 0;
    reverse(vSortedByTimestamp.begin(), vSortedByTimestamp.end());
    sort(vSortedByTimestamp.begin(), vSortedByTimestamp.end());

    // Select 64 blocks from candidate blocks to generate stake modifier
    uint64_t nStakeModifierNew = 0;
    int64_t nSelectionIntervalStop = nSelectionIntervalStart;
    map<uint256, const CBlockIndex*> mapSelectedBlocks;
    for (int nRound=0; nRound<min(64, (int)vSortedByTimestamp.size()); nRound++)
    {
        // add an interval section to the current selection round
        nSelectionIntervalStop += GetStakeModifierSelectionIntervalSection(nRound);
        // select a block from the candidates of current round
        if (!SelectBlockFromCandidates(vSortedByTimestamp, mapSelectedBlocks, nSelectionIntervalStop, nStakeModifier, &pindex))
            return error("ComputeNextStakeModifier: unable to select block at round %d", nRound);
        // write the entropy bit of the selected block
        nStakeModifierNew |= (((uint64_t)pindex->GetStakeEntropyBit()) << nRound);
        // add the selected block from candidates to selected list
        mapSelectedBlocks.insert(make_pair(pindex->GetBlockHash(), pindex));
        if (fDebug && gArgs.GetBoolArg("-printstakemodifier"))
          LogPrintf(
              "ComputeNextStakeModifier: selected round %d stop=%s height=%d "
              "bit=%d\n",
              nRound, DateTimeStrFormat(nSelectionIntervalStop),
              pindex->nHeight, pindex->GetStakeEntropyBit());
    }

    // Print selection map for visualization of the selected blocks
    if (fDebug && gArgs.GetBoolArg("-printstakemodifier"))
    {
        string strSelectionMap = "";
        // '-' indicates proof-of-work blocks not selected
        strSelectionMap.insert(0, pindexPrev->nHeight - nHeightFirstCandidate + 1, '-');
        pindex = pindexPrev;
        while (pindex && pindex->nHeight >= nHeightFirstCandidate)
        {
            // '=' indicates proof-of-stake blocks not selected
            if (pindex->IsProofOfStake())
                strSelectionMap.replace(pindex->nHeight - nHeightFirstCandidate, 1, "=");
            pindex = pindex->pprev;
        }
        BOOST_FOREACH(const PAIRTYPE(uint256, const CBlockIndex*)& item, mapSelectedBlocks)
        {
            // 'S' indicates selected proof-of-stake blocks
            // 'W' indicates selected proof-of-work blocks
            strSelectionMap.replace(item.second->nHeight - nHeightFirstCandidate, 1, item.second->IsProofOfStake()? "S" : "W");
        }
        LogPrintf(
            "ComputeNextStakeModifier: selection height [%d, %d] map %s\n",
            nHeightFirstCandidate, pindexPrev->nHeight,
            strSelectionMap);
    }
    if (fDebug)
    {
      LogPrintf("ComputeNextStakeModifier: new modifier=0x%016" PRIx64
                " time=%s\n",
                nStakeModifierNew,
                DateTimeStrFormat(pindexPrev->GetBlockTime()));
    }

    nStakeModifier = nStakeModifierNew;
    fGeneratedStakeModifier = true;
    return true;
}

// The stake modifier used to hash for a stake kernel is chosen as the stake
// modifier about a selection interval later than the coin generating the kernel
static bool GetKernelStakeModifier(CBlockIndex* pindexPrev, uint256 hashBlockFrom, uint64_t& nStakeModifier, int& nStakeModifierHeight, int64_t& nStakeModifierTime, bool fPrintProofOfStake)
{
    const Consensus::Params& params = Params().GetConsensus();
    nStakeModifier = 0;
    if (!mapBlockIndex.count(hashBlockFrom))
        return error("GetKernelStakeModifier() : block not indexed");
    const CBlockIndex* pindexFrom = mapBlockIndex[hashBlockFrom];
    nStakeModifierHeight = pindexFrom->nHeight;
    nStakeModifierTime = pindexFrom->GetBlockTime();
    int64_t nStakeModifierSelectionInterval = GetStakeModifierSelectionInterval();

    // we need to iterate index forward but we cannot depend on chainActive.Next()
    // because there is no guarantee that we are checking blocks in active chain.
    // So, we construct a temporary chain that we will iterate over.
    // pindexFrom - this block contains coins that are used to generate PoS
    // pindexPrev - this is a block that is previous to PoS block that we are checking, you can think of it as tip of our chain
    std::vector<CBlockIndex*> tmpChain;
    int32_t nDepth = pindexPrev->nHeight - (pindexFrom->nHeight-1); // -1 is used to also include pindexFrom
    tmpChain.reserve(nDepth);
    CBlockIndex* it = pindexPrev;
    for (int i=1; i<=nDepth && !chainActive.Contains(it); i++) {
        tmpChain.push_back(it);
        it = it->pprev;
    }
    std::reverse(tmpChain.begin(), tmpChain.end());
    size_t n = 0;

    const CBlockIndex* pindex = pindexFrom;
    // loop to find the stake modifier later by a selection interval
    while (nStakeModifierTime < pindexFrom->GetBlockTime() + nStakeModifierSelectionInterval)
    {
        const CBlockIndex* old_pindex = pindex;
        pindex = (!tmpChain.empty() && pindex->nHeight >= tmpChain[0]->nHeight - 1)? tmpChain[n++] : chainActive.Next(pindex);
        if (n > tmpChain.size() || pindex == NULL) // check if tmpChain[n+1] exists
        {   // reached best block; may happen if node is behind on block chain
            if (fPrintProofOfStake || (old_pindex->GetBlockTime() + params.nStakeMinAge - nStakeModifierSelectionInterval > GetAdjustedTime()))
                return error("GetKernelStakeModifier() : reached best block %s at height %d from block %s",
                    old_pindex->GetBlockHash().ToString(), old_pindex->nHeight, hashBlockFrom.ToString());
            else
                return false;
        }
        if (pindex->GeneratedStakeModifier())
        {
            nStakeModifierHeight = pindex->nHeight;
            nStakeModifierTime = pindex->GetBlockTime();
        }
    }
    nStakeModifier = pindex->nStakeModifier;
    return true;
}

// yacoin2015
uint256 GetProofOfStakeHash( 
                            uint64_t nStakeModifier, 
                            uint32_t nTimeBlockFrom, 
                            uint32_t nTxPrevOffset, 
                            uint32_t nTxPrevTime, 
                            uint32_t nPrevoutn, 
                            uint32_t nTimeTx, 
                            uint64_t nCoinDayWeight 
                           )
{
    uint256 thash;
    CDataStream
        ss(SER_GETHASH, 0);

    ss << nStakeModifier;
    ss << nTimeBlockFrom << nTxPrevOffset << nTxPrevTime << nPrevoutn << nTimeTx;
    thash = Hash(ss.begin(), ss.end());
    return thash;
}

//_____________________________________________________________________________
// ppcoin kernel protocol
// coinstake must meet hash target according to the protocol:
// kernel (input 0) must meet the formula
//     hash(
//          nStakeModifier + 
//          txPrev.block.nTime + 
//          txPrev.offset + 
//          txPrev.nTime + 
//          txPrev.vout.n + 
//          nTime
//         ) < (bnTarget * nCoinDayWeight)
// this ensures that the chance of getting a coinstake is proportional to the
// amount of coin age one owns.
// The reason this hash is chosen is the following:
//   nStakeModifier: scrambles computation to make it very difficult to precompute
//                  future proof-of-stake at the time of the coin's confirmation
//   txPrev.block.nTime: prevent nodes from guessing a good timestamp to
//                       generate transaction for future advantage
//   txPrev.offset: offset of txPrev inside block, to reduce the chance of 
//                  nodes generating coinstake at the same time
//   txPrev.nTime: reduce the chance of nodes generating coinstake at the same
//                 time
//   txPrev.vout.n: output number of txPrev, to reduce the chance of nodes
//                  generating coinstake at the same time
//   block/tx hash should not be used here as they can be generated in vast
//   quantities so as to generate blocks faster, degrading the system back into
//   a proof-of-work situation.
//
bool CheckStakeKernelHash(uint32_t nBits, CBlockIndex* pindexPrev,
                          const CBlockHeader& blockFrom, uint32_t nTxPrevOffset,
                          const CTransaction& txPrev, const COutPoint& prevout,
                          uint32_t nTimeTx, uint256& hashProofOfStake,
                          uint256& targetProofOfStake,
                          bool fPrintProofOfStake)
{
    const Consensus::Params& params = Params().GetConsensus();
    if ((::int64_t)nTimeTx < txPrev.nTime)  // Transaction timestamp violation
        return error("CheckStakeKernelHash () : nTime violation");

    uint32_t nTimeBlockFrom = (::uint64_t)blockFrom.GetBlockTime();
    if (nTimeBlockFrom + params.nStakeMinAge > nTimeTx) // Min age requirement
        return error("CheckStakeKernelHash () : min age violation");

    CBigNum bnTargetPerCoinDay;
    bnTargetPerCoinDay.SetCompact(nBits);
    ::int64_t nValueIn = txPrev.vout[prevout.COutPointGet_n()].nValue;

    ::int64_t
        nTimeWeight = min((::int64_t)nTimeTx - txPrev.nTime, (::int64_t)params.nStakeMaxAge) - params.nStakeMinAge;
    CBigNum 
        bnCoinDayWeight = CBigNum(nValueIn) * GetWeight(
                                                        (int64_t)txPrev.nTime, 
                                                        (int64_t)nTimeTx
                                                       ) / COIN / (24 * 60 * 60);
    //targetProofOfStake = (bnCoinDayWeight * bnTargetPerCoinDay).getuint256();

    // Calculate hash
    uint256 hashBlockFrom = blockFrom.GetHash();

    uint64_t nStakeModifier = 0;
    int nStakeModifierHeight = 0;
    ::int64_t nStakeModifierTime = 0;

    if (!GetKernelStakeModifier(pindexPrev, hashBlockFrom, nStakeModifier, nStakeModifierHeight, nStakeModifierTime, fPrintProofOfStake))
        return false;

    // yacoin2015: scrypt-jane hash for stake kernel
    hashProofOfStake = GetProofOfStakeHash( 
                                           nStakeModifier, 
                                           nTimeBlockFrom, 
                                           nTxPrevOffset, 
                                           txPrev.nTime, 
                                           prevout.COutPointGet_n(), 
                                           nTimeTx, 
                                           bnCoinDayWeight.getuint64() 
                                          );

    if (fPrintProofOfStake)
    {
        LogPrintf("CheckStakeKernelHash (a) : using modifier 0x%016" PRIx64
            "\n"
            " at height=%d"
            " timestamp=%s"
            "\n"
            " for block from height=%d"
            " timestamp=%s"
            "\n"
            , nStakeModifier
            , nStakeModifierHeight
            , DateTimeStrFormat(nStakeModifierTime)
            , mapBlockIndex[hashBlockFrom]->nHeight
            , DateTimeStrFormat(blockFrom.GetBlockTime())
              );
        LogPrintf(
            "CheckStakeKernelHash (a) : check modifier=0x%016" PRIx64 
            "\n"
            " nTimeBlockFrom=%u"
            " nTxPrevOffset=%u"
            "\n"
            " nTimeTxPrev=%" PRId64
            "\n"
            " nPrevout=%u"
            " nTimeTx=%u"
            "\n"
            " hashProof=%s"
            "\n"
            , nStakeModifier
            , nTimeBlockFrom
            , nTxPrevOffset
            , txPrev.nTime
            , prevout.COutPointGet_n()
            , nTimeTx
            , hashProofOfStake.ToString()
              );
    }

    // Now check if proof-of-stake hash meets target protocol
    if (
        (CBigNum(hashProofOfStake) > (bnCoinDayWeight * bnTargetPerCoinDay))
       )
        return false;
    if (fDebug && !fPrintProofOfStake)
    {
        LogPrintf(
            "CheckStakeKernelHash (b) : using modifier 0x%016" PRIx64 " "
            "\n"
            "at height=%d "
            "timestamp=%s "
            "\n"
            "for block from height=%d "
            "timestamp=%s"
            "\n"
            , nStakeModifier
            , nStakeModifierHeight
            , DateTimeStrFormat(nStakeModifierTime)
            , mapBlockIndex[hashBlockFrom]->nHeight
            , DateTimeStrFormat(blockFrom.GetBlockTime())
              );
        LogPrintf(
            "CheckStakeKernelHash (b) : pass modifier=0x%016" PRIx64 " "
            "\n"
            "nTimeBlockFrom=%u "
            "nTxPrevOffset=%u "
            "\n"
            "nTimeTxPrev=%" PRIu64 " "
            "\n"
            "nPrevout=%u "
            "nTimeTx=%u "
            "\n"
            "hashProof=%s"
            "\n"
            , nStakeModifier
            , nTimeBlockFrom
            , nTxPrevOffset
            , txPrev.nTime
            , prevout.COutPointGet_n()
            , nTimeTx
            , hashProofOfStake.ToString()
              );
    }
    targetProofOfStake = (bnCoinDayWeight * bnTargetPerCoinDay).getuint256();
    return true;
}

// Check kernel hash target and coinstake signature
bool CheckProofOfStake(CValidationState &state, CBlockIndex* pindexPrev, const CTransaction& tx, unsigned int nBits, uint256& hashProofOfStake, uint256& targetProofOfStake)
{
    if (!tx.IsCoinStake())
        return error("CheckProofOfStake() : called on non-coinstake %s", tx.GetHash().ToString().c_str());

    // Kernel (input 0) must match the stake hash target per coin age (nBits)
    const CTxIn& txin = tx.vin[0];

    // Transaction index is required to get to block header
    if (!fTxIndex)
        return error("CheckProofOfStake() : transaction index not available");

    // Get transaction index for the previous transaction
    CDiskTxPos postx;
    if (!pblocktree->ReadTxIndex(txin.prevout.hash, postx))
        return error("CheckProofOfStake() : tx index not found");  // tx index not found

    // Read txPrev and header of its block
    CBlockHeader header;
    CTransaction txPrev;
    {
        CAutoFile file(OpenBlockFile(postx, true), SER_DISK, CLIENT_VERSION);
        try {
            file >> header;
            fseek(file.Get(), postx.nTxOffset, SEEK_CUR);
            file >> txPrev;
        } catch (std::exception &e) {
            return error("%s() : deserialize or I/O error in CheckProofOfStake()", __PRETTY_FUNCTION__);
        }
        if (txPrev.GetHash() != txin.prevout.hash)
            return error("%s() : txid mismatch in CheckProofOfStake()", __PRETTY_FUNCTION__);
    }

    // Retrieve blockhash to avoid recalculating block hash (very slow !!!)
    if (fBlockHashIndex && !pblocktree->ReadBlockHash(postx.nFile, postx.nPos, header.blockHash)) {
        LogPrintf("CheckProofOfStake(): can't read block hash at file = %d, block pos = %d\n", postx.nFile, postx.nPos);
    }

    // Verify signature
    {
        int nIn = 0;
        const CTxOut& prevOut = txPrev.vout[tx.vin[nIn].prevout.n];
        TransactionSignatureChecker checker(&tx, nIn);

        if (!VerifyScript(tx.vin[nIn].scriptSig, prevOut.scriptPubKey, MANDATORY_SCRIPT_VERIFY_FLAGS, checker, nullptr))
            return state.DoS(100, error("CheckProofOfStake() : VerifySignature failed on coinstake %s", tx.GetHash().ToString().c_str()));
    }

    if (!CheckStakeKernelHash(nBits, pindexPrev, header, postx.nTxOffset + ::GetSerializeSize(header, SER_DISK, CLIENT_VERSION), txPrev, txin.prevout, tx.nTime, hashProofOfStake, targetProofOfStake, gArgs.GetBoolArg("-debug", false)))
        return state.DoS(1, error("CheckProofOfStake() : INFO: check kernel failed on coinstake %s, hashProof=%s", tx.GetHash().ToString(), hashProofOfStake.ToString())); // may occur during initial download or if behind on block chain sync

    return true;
}

// Get stake modifier checksum
uint32_t GetStakeModifierChecksum(const CBlockIndex* pindex)
{
    if( 0 < pindex->nHeight )
    {
        Yassert( pindex->pprev );
    }
    if( 0 == pindex->nHeight )
    {
        Yassert( pindex->GetBlockHash() == Params().GetConsensus().hashGenesisBlock);
    }
    // Hash previous checksum with flags, hashProofOfStake and nStakeModifier
    CDataStream ss(SER_GETHASH, 0);
    if (pindex->pprev)
        ss << pindex->pprev->nStakeModifierChecksum;
    ss << pindex->nFlags << pindex->hashProofOfStake << pindex->nStakeModifier;
    uint256 hashChecksum = Hash(ss.begin(), ss.end());
    hashChecksum >>= (256 - 32);
    return hashChecksum.Get64();
}

// Check stake modifier hard checkpoints
bool CheckStakeModifierCheckpoints(int nHeight, uint32_t nStakeModifierChecksum)
{
    if (chainActive.Tip() && (chainActive.Tip()->nHeight + 1) >= nMainnetNewLogicBlockNumber)
    {
        return true;
    }

    MapModifierCheckpoints& checkpoints = (fTestNet ? mapStakeModifierCheckpointsTestNet : mapStakeModifierCheckpoints);

    if (checkpoints.count(nHeight))
        return nStakeModifierChecksum == checkpoints[nHeight];

    return true;
}
#ifdef _MSC_VER
    #include "msvc_warnings.pop.h"
#endif
