// =============================================================================
// paver_runtime.h - PAVER TB-group runtime for GPGPU-Sim
// =============================================================================
//
// Implements the runtime of D. Tripathy et al., "PAVER: Locality Graph-Based
// Thread Block Scheduling for GPUs", ACM TACO 18(3), 2021, Section 5:
//
//  * Fig. 8  - the TB groups produced at JIT time live in a global table. Each
//              SM has next/tail registers that point into the TB group it
//              runs (64-bit pointers in the paper; a group index plus two
//              offsets here) and a nextTB register. A TB is issued "as soon
//              as a free TB context is available" (Sec. 5.3).
//  * Fig. 9  - when SM.next == SM.tail the SM moves on to its next TB group;
//              when it has none left it becomes a task-stealing recipient.
//              In the paper, groups not yet handed out wait in a global queue
//              and go to whichever SM exhausts its group first. Here the groups
//              of every SM are fixed in advance by the config
//              (paver_sm_assign). The two are the same when every SM has at
//              most one TB group, which is the case for K-Way-TS (k = #SMs).
//              Configs that give an SM several groups (MST-TS, RB-TS, TSP-TS)
//              get a static round-robin hand-out instead of the dynamic queue.
//  * Alg. 2  - WaitingTB = SM.tail - SM.next, DonorSM = SM with the largest
//              WaitingTB, StolenTBcount = MaxWaitingTB - AverageWaitingTB
//              (average over all SMs). The stolen TBs are the last
//              StolenTBcount TBs of the donor's group: recipient.tail =
//              donor.tail, donor.tail -= StolenTBcount, recipient.next =
//              donor.tail. (Fig. 9 writes donor.tail + 1 because its tail is
//              inclusive. The example in Sec. 5.3 calls that register
//              "Recipient SM.tail", which contradicts its own first step;
//              Fig. 9 has SM.next.) waiting() explains how WaitingTB is
//              adapted to GPGPU-Sim's issue rate.
//
// This header has no GPGPU-Sim dependencies so it can be unit-tested alone.
// Ranges are half-open: an SM still has (tail - next) unissued TBs.
// =============================================================================

#ifndef PAVER_RUNTIME_H
#define PAVER_RUNTIME_H

#include <cstdio>
#include <map>
#include <vector>

class paver_runtime {
 public:
  paver_runtime()
      : m_enabled(false),
        m_task_stealing(false),
        m_num_sms(0),
        m_total_unissued(0),
        m_issued(0),
        m_steal_events(0),
        m_stolen_tbs(0),
        m_diagnosed(false) {}

  // groups:         TB groups (linear CTA ids, issue order). Index == the
  //                 'paver_partition <i>' line of paver_config.txt.
  // sm_assignments: GPGPU-Sim SM id -> indices of the TB groups that SM runs,
  //                 in order.
  // num_sms:        number of SMs in the simulated GPU.
  // The caller validates the inputs (paver_config_loader::validate).
  void init(const std::vector<std::vector<unsigned> > &groups,
            const std::map<unsigned, std::vector<unsigned> > &sm_assignments,
            bool task_stealing, unsigned num_sms) {
    m_groups = groups;
    m_num_sms = num_sms;
    m_assigned.assign(num_sms, std::vector<unsigned>());
    for (std::map<unsigned, std::vector<unsigned> >::const_iterator it =
             sm_assignments.begin();
         it != sm_assignments.end(); ++it)
      if (it->first < num_sms) m_assigned[it->first] = it->second;
    m_sm.assign(num_sms, sm_regs());
    m_total_unissued = 0;
    // Kernel launch: every SM's registers point at its first TB group.
    for (unsigned sm = 0; sm < num_sms; sm++) load_next_group(sm);
    m_task_stealing = task_stealing;
    m_issued = m_steal_events = m_stolen_tbs = 0;
    m_diagnosed = false;
    m_enabled = true;
  }

  bool enabled() const { return m_enabled; }
  bool task_stealing() const { return m_task_stealing; }

  // Linear id of the next TB SM `sm` should issue, or -1 if it has none.
  // Call only when the SM has a free TB context for this kernel;
  // max_cta_per_sm is how many CTAs of this kernel an SM can hold.
  int next_tb(unsigned sm, unsigned max_cta_per_sm) {
    if (!m_enabled || sm >= m_num_sms) return -1;
    if (!m_diagnosed) diagnose(max_cta_per_sm);
    sm_regs &r = m_sm[sm];
    while (r.next >= r.tail) {  // SM.next == SM.tail: TB group exhausted
      if (load_next_group(sm)) continue;
      if (!m_task_stealing || !steal(sm, max_cta_per_sm)) return -1;
    }
    m_total_unissued--;
    return (int)m_groups[r.group][r.next++];
  }

  void cta_issued(unsigned sm) {
    if (sm < m_num_sms) m_sm[sm].running++;
    m_issued++;
  }
  void cta_completed(unsigned sm) {
    if (sm < m_num_sms && m_sm[sm].running > 0) m_sm[sm].running--;
  }

  unsigned issued() const { return m_issued; }
  unsigned steal_events() const { return m_steal_events; }
  unsigned stolen_tbs() const { return m_stolen_tbs; }

 private:
  struct sm_regs {
    int group;            // TB group next/tail point into (-1: none)
    unsigned next;        // index of the next TB to issue in that group
    unsigned tail;        // one past the last TB of that group owned by the SM
    unsigned assign_pos;  // next entry of this SM's assigned-group list
    unsigned running;     // CTAs of this kernel running on the SM
    sm_regs() : group(-1), next(0), tail(0), assign_pos(0), running(0) {}
  };

  // Point the SM's registers at its next non-empty assigned TB group.
  bool load_next_group(unsigned sm) {
    sm_regs &r = m_sm[sm];
    const std::vector<unsigned> &assigned = m_assigned[sm];
    while (r.assign_pos < assigned.size()) {
      unsigned g = assigned[r.assign_pos++];
      if (g < m_groups.size() && !m_groups[g].empty()) {
        r.group = (int)g;
        r.next = 0;
        r.tail = (unsigned)m_groups[g].size();
        m_total_unissued += r.tail;
        return true;
      }
    }
    r.group = -1;
    r.next = r.tail = 0;
    return false;
  }

  // WaitingTB of Algorithm 2. The paper computes SM.tail - SM.next. On the
  // PAVER hardware an SM issues SM.nextTB as soon as a TB context is free, so
  // at steal time every one of those TBs is waiting for a context. GPGPU-Sim
  // issues at most one CTA per cluster per cycle, so an SM can briefly hold
  // unissued TBs and free contexts at the same time (e.g. at kernel launch);
  // the TBs about to fill those contexts are not waiting. This returns
  // (tail - next) minus the SM's free contexts, which equals tail - next
  // whenever the SM is full.
  unsigned waiting(unsigned sm, unsigned max_cta_per_sm) const {
    const sm_regs &r = m_sm[sm];
    unsigned unissued = r.tail - r.next;
    unsigned free_ctx =
        r.running < max_cta_per_sm ? max_cta_per_sm - r.running : 0;
    return unissued > free_ctx ? unissued - free_ctx : 0;
  }

  // Algorithm 2 + Fig. 9. Returns true if TBs were moved to `recipient`.
  bool steal(unsigned recipient, unsigned max_cta_per_sm) {
    if (m_total_unissued == 0) return false;
    unsigned max_waiting = 0, sum_waiting = 0;
    int donor = -1;
    for (unsigned sm = 0; sm < m_num_sms; sm++) {
      unsigned w = waiting(sm, max_cta_per_sm);
      if (w > max_waiting) {
        max_waiting = w;
        donor = (int)sm;
      }
      sum_waiting += w;
    }
    if (donor < 0 || (unsigned)donor == recipient) return false;
    unsigned average_waiting = sum_waiting / m_num_sms;
    // 1 <= count <= max_waiting: the recipient has no unissued TBs, so the
    // average is below the maximum; and the donor keeps at least as many
    // unissued TBs as it has free contexts.
    unsigned count = max_waiting - average_waiting;
    if (count == 0) return false;

    sm_regs &d = m_sm[donor];
    sm_regs &r = m_sm[recipient];
    r.group = d.group;  // recipient.tail = donor.tail
    r.tail = d.tail;
    d.tail -= count;    // donor.tail -= StolenTBcount
    r.next = d.tail;    // recipient.next = donor.tail (+1, inclusive form)
    m_steal_events++;
    m_stolen_tbs += count;
    return true;
  }

  // One-time check: an SM that owns more TBs than it can run at once while
  // other SMs own none executes them in serial waves. For grids with fewer
  // TBs than the GPU can hold at once, shader_core_config::max_cta() lowers
  // the per-SM limit to ceil(#TBs / #SMs), e.g. 1 for 16 TBs on 80 SMs.
  void diagnose(unsigned max_cta_per_sm) {
    m_diagnosed = true;
    unsigned idle = 0, worst_sm = 0, worst = 0;
    for (unsigned sm = 0; sm < m_num_sms; sm++) {
      unsigned owned = 0;
      for (size_t i = 0; i < m_assigned[sm].size(); i++)
        if (m_assigned[sm][i] < m_groups.size())
          owned += (unsigned)m_groups[m_assigned[sm][i]].size();
      if (owned == 0) idle++;
      if (owned > worst) {
        worst = owned;
        worst_sm = sm;
      }
    }
    if (idle > 0 && max_cta_per_sm > 0 && worst > max_cta_per_sm)
      printf(
          "PAVER: WARNING: SM %u owns %u TBs but can run %u at a time, while "
          "%u SMs own none; its TBs run in %u serial waves%s.\n",
          worst_sm, worst, max_cta_per_sm, idle,
          (worst + max_cta_per_sm - 1) / max_cta_per_sm,
          m_task_stealing ? " unless task stealing moves them"
                          : " (task stealing is off)");
  }

  bool m_enabled;
  bool m_task_stealing;
  unsigned m_num_sms;
  std::vector<std::vector<unsigned> > m_groups;    // global TB-group table
  std::vector<std::vector<unsigned> > m_assigned;  // per-SM group indices
  std::vector<sm_regs> m_sm;                       // per-SM registers
  unsigned m_total_unissued;  // sum over SMs of (tail - next)
  unsigned m_issued;
  unsigned m_steal_events;
  unsigned m_stolen_tbs;
  bool m_diagnosed;
};

#endif  // PAVER_RUNTIME_H
