// =============================================================================
// paver_config_loader.h
// =============================================================================
// Reads a PAVER scheduler config produced by LocalityGuru_2
// (paver_runtime_integration.py, SchedulerConfig.to_gpgpu_sim_format):
//
//   paver_enabled 1
//   paver_method kway_ts
//   paver_task_stealing 1
//   paver_sm_order linear|interleaved   optional, default linear (see below)
//   paver_partition <i> <tb>,<tb>,...   TB group <i>, TBs in issue order
//   paver_sm_assign <sm> <i>,<i>,...    TB groups executed by SM <sm>
//
// Other keywords (paver_num_partitions, paver_tb_schedule) are informational.
// Numbers must be plain unsigned decimals: "-1", "+5", "12abc" and values
// above 4294967295 are rejected instead of being wrapped or truncated.
// TB group ids must be 0..N-1, each defined once (the exporter numbers them
// that way), and an SM may appear in only one paver_sm_assign line. A file
// that breaks these rules is ignored with a message, so memory use is bounded
// by the size of the file, whatever ids it contains.
// =============================================================================

#ifndef PAVER_CONFIG_LOADER_H
#define PAVER_CONFIG_LOADER_H

#include <cstdio>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class paver_config_loader {
 public:
  bool enabled;
  std::string method;
  bool task_stealing;
  std::string sm_order;  // "linear" or "interleaved"
  std::vector<std::vector<unsigned> > partitions;
  std::map<unsigned, std::vector<unsigned> > sm_assignments;

  paver_config_loader()
      : enabled(false), task_stealing(true), sm_order("linear") {}

  // Returns true if the file was read and contains 'paver_enabled 1'.
  bool load(const std::string &filepath) {
    enabled = false;
    method.clear();
    task_stealing = true;
    sm_order = "linear";
    partitions.clear();
    sm_assignments.clear();

    std::ifstream file(filepath.c_str());
    if (!file.is_open()) return false;

    // TB groups by id; copied into `partitions` once the ids are known to be
    // 0..N-1, so a corrupt id cannot make the table larger than the file.
    std::map<unsigned, std::vector<unsigned> > groups;
    std::string line;
    unsigned lineno = 0;
    try {
      while (std::getline(file, line)) {
        lineno++;
        if (!line.empty() && line[line.size() - 1] == '\r')
          line.erase(line.size() - 1);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "paver_enabled") {
          int val = 0;
          iss >> val;
          enabled = (val == 1);
        } else if (keyword == "paver_method") {
          iss >> method;
        } else if (keyword == "paver_task_stealing") {
          int val = 1;
          iss >> val;
          task_stealing = (val == 1);
        } else if (keyword == "paver_sm_order") {
          std::string v = next_token(iss);
          if (v != "linear" && v != "interleaved")
            throw std::invalid_argument(
                "paver_sm_order must be 'linear' or 'interleaved'");
          sm_order = v;
        } else if (keyword == "paver_partition") {
          unsigned id = parse_uint(next_token(iss));
          if (groups.count(id))
            throw std::invalid_argument("TB group " + to_str(id) +
                                        " is defined twice");
          groups[id] = parse_list(next_token(iss));
        } else if (keyword == "paver_sm_assign") {
          unsigned sm = parse_uint(next_token(iss));
          if (sm_assignments.count(sm))
            throw std::invalid_argument("SM " + to_str(sm) +
                                        " is assigned twice");
          sm_assignments[sm] = parse_list(next_token(iss));
        }
      }
    } catch (const std::exception &e) {
      printf("PAVER: %s:%u: cannot parse \"%s\" (%s); config ignored\n",
             filepath.c_str(), lineno, line.c_str(), e.what());
      enabled = false;
      partitions.clear();
      sm_assignments.clear();
      return false;
    }

    // Ids are unique, so they are exactly 0..N-1 iff the largest is N-1.
    if (!groups.empty() && groups.rbegin()->first != groups.size() - 1) {
      printf(
          "PAVER: %s: TB group ids must be 0..N-1 (N = %zu here), but the "
          "largest id is %u; config ignored\n",
          filepath.c_str(), groups.size(), groups.rbegin()->first);
      enabled = false;
      sm_assignments.clear();
      return false;
    }
    for (std::map<unsigned, std::vector<unsigned> >::iterator it =
             groups.begin();
         it != groups.end(); ++it)
      partitions.push_back(it->second);

    if (enabled)
      printf(
          "PAVER: Loaded %s - method=%s, TB groups=%zu, task_stealing=%d, "
          "sm_order=%s\n",
          filepath.c_str(), method.c_str(), partitions.size(), task_stealing,
          sm_order.c_str());
    return enabled;
  }

  // A config is usable for a kernel only if, through sm_assignments, every
  // TB of the grid is assigned exactly once, to an SM that exists. Otherwise
  // PAVER would issue CTAs outside the grid, issue some twice, or never
  // finish the kernel.
  bool validate(size_t num_blocks, unsigned num_sms, std::string &err) const {
    std::ostringstream os;
    std::vector<unsigned char> seen(num_blocks, 0);
    for (std::map<unsigned, std::vector<unsigned> >::const_iterator it =
             sm_assignments.begin();
         it != sm_assignments.end(); ++it) {
      if (it->second.empty()) continue;
      if (it->first >= num_sms) {
        os << "SM " << it->first << " is assigned TB groups but the GPU has "
           << num_sms << " SMs";
        err = os.str();
        return false;
      }
      for (size_t i = 0; i < it->second.size(); i++) {
        unsigned g = it->second[i];
        if (g >= partitions.size()) {
          os << "SM " << it->first << " uses TB group " << g << " but only "
             << partitions.size() << " groups are defined";
          err = os.str();
          return false;
        }
        for (size_t j = 0; j < partitions[g].size(); j++) {
          unsigned tb = partitions[g][j];
          if (tb >= num_blocks) {
            os << "TB " << tb << " (group " << g << ") is outside the grid of "
               << num_blocks << " TBs";
            err = os.str();
            return false;
          }
          if (seen[tb]) {
            os << "TB " << tb << " is assigned more than once";
            err = os.str();
            return false;
          }
          seen[tb] = 1;
        }
      }
    }
    size_t missing = 0, first = 0;
    for (size_t tb = 0; tb < num_blocks; tb++)
      if (!seen[tb] && missing++ == 0) first = tb;
    if (missing) {
      os << missing << " of " << num_blocks
         << " TBs are not assigned to any SM (first: TB " << first << ")";
      err = os.str();
      return false;
    }
    return true;
  }

  // sm_assignments with the config's SM numbers turned into GPGPU-Sim SM ids
  // (sid = cluster * cores_per_cluster + core, as cid_to_sid() computes).
  //   linear:      config SM i is GPGPU-Sim SM i (what earlier versions did).
  //   interleaved: config SM i is the i-th SM the stock CTA dispatcher fills
  //                when it starts at cluster 0, as it does for the first
  //                kernel of a run: core 0 of clusters 0..C-1, then core 1 of
  //                clusters 0..C-1, ... (it visits clusters round-robin and
  //                issues at most one CTA per cluster per cycle). A config
  //                that uses only config SMs 0..m-1 (m <= C) then occupies one
  //                SM in each of m clusters, as the baseline does for an m-TB
  //                kernel, instead of both SMs of m/2 clusters. For later
  //                kernels the baseline starts after the last cluster it used
  //                (m_last_cluster_issue is not reset), so it picks other SMs,
  //                but still one per cluster.
  // Call after validate(); entries for SMs that do not exist (allowed only
  // with an empty group list) are dropped.
  std::map<unsigned, std::vector<unsigned> > physical_sm_assignments(
      unsigned n_clusters, unsigned cores_per_cluster) const {
    if (sm_order != "interleaved") return sm_assignments;
    const unsigned num_sms = n_clusters * cores_per_cluster;
    std::map<unsigned, std::vector<unsigned> > out;
    for (std::map<unsigned, std::vector<unsigned> >::const_iterator it =
             sm_assignments.begin();
         it != sm_assignments.end(); ++it) {
      if (it->first >= num_sms) continue;
      unsigned cluster = it->first % n_clusters;
      unsigned core = it->first / n_clusters;
      out[cluster * cores_per_cluster + core] = it->second;
    }
    return out;
  }

 private:
  static std::string next_token(std::istringstream &iss) {
    std::string tok;
    iss >> tok;
    return tok;
  }
  static std::string to_str(unsigned v) {
    std::ostringstream os;
    os << v;
    return os.str();
  }
  static unsigned parse_uint(const std::string &s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("not an unsigned integer: \"" + s + "\"");
    unsigned long long v = std::stoull(s);  // out_of_range if too long
    if (v > 0xFFFFFFFFull)
      throw std::out_of_range("number too large: " + s);
    return (unsigned)v;
  }
  static std::vector<unsigned> parse_list(const std::string &s) {
    std::vector<unsigned> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
      if (!tok.empty()) out.push_back(parse_uint(tok));
    return out;
  }
};

#endif  // PAVER_CONFIG_LOADER_H
