#ifndef ANALYSIS_HPP
#define ANALYSIS_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>

/**
 * @brief Computes the Mean Squared Displacement (MSD) of particles
 *
 * MSD(t) = ⟨|x(t) - x(0)|²⟩
 *
 * This measures how far particles diffuse over time. For normal Brownian motion,
 * MSD grows linearly with time: MSD ∝ t.
 *
 * @param x Current x positions of particles
 * @param y Current y positions of particles
 * @param x0 Initial x positions of particles
 * @param y0 Initial y positions of particles
 * @param domain_size Size of the simulation domain (for periodic boundaries)
 * @return Mean Squared Displacement (MSD)
 */
inline float compute_msd(const std::vector<float>& x, const std::vector<float>& y,
                         const std::vector<float>& x0, const std::vector<float>& y0,
                         float domain_size) {
    size_t N = x.size();
    float msd = 0.0f;
    const float half_domain = domain_size / 2.0f;
    
    for (size_t i = 0; i < N; ++i) {
        // Compute displacement with minimum image convention for periodic boundaries
        float dx = x[i] - x0[i];
        if (dx > half_domain) dx -= domain_size;
        else if (dx < -half_domain) dx += domain_size;

        float dy = y[i] - y0[i];
        if (dy > half_domain) dy -= domain_size;
        else if (dy < -half_domain) dy += domain_size;

        msd += dx * dx + dy * dy;
    }

    return msd / static_cast<float>(N);
}

/**
 * @brief Save MSD data to CSV file for analysis
 *
 * Creates a CSV file with columns: time,msd
 * This can be loaded by Python/MATLAB for plotting and analysis.
 *
 * @param time_points Vector of time values
 * @param msd_values Vector of corresponding MSD values
 * @param filename Output filename (e.g., "output/msd_data.csv")
 * @return true if successful, false if file write failed
 */
inline bool save_msd_to_csv(const std::vector<float>& time_points,
                            const std::vector<float>& msd_values,
                            const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << "\n";
        return false;
    }
    
    // Write header
    file << "time,msd\n";
    
    // Write data
    for (size_t i = 0; i < time_points.size(); ++i) {
        file << time_points[i] << "," << msd_values[i] << "\n";
    }
    
    file.close();
    return true;
}

/**
 * @brief Calculate statistics for a vector of values
 */
struct Stats {
    float mean;
    float stddev;
    float min_val;
    float max_val;
};

inline Stats calculate_stats(const std::vector<float>& data) {
    Stats s;
    if (data.empty()) {
        s.mean = s.stddev = s.min_val = s.max_val = 0.0f;
        return s;
    }
    
    s.min_val = data[0];
    s.max_val = data[0];
    float sum = 0.0f;
    
    for (float val : data) {
        sum += val;
        if (val < s.min_val) s.min_val = val;
        if (val > s.max_val) s.max_val = val;
    }
    
    s.mean = sum / data.size();
    
    float sq_sum = 0.0f;
    for (float val : data) {
        sq_sum += (val - s.mean) * (val - s.mean);
    }
    s.stddev = std::sqrt(sq_sum / data.size());
    
    return s;
}

/**
 * @brief Compute the Velocity Autocorrelation Function (VACF)
 *
 * VACF(τ) = ⟨v(t) · v(t+τ)⟩ / ⟨v²⟩
 *
 * Measures temporal correlation of velocity. VACF decays exponentially
 * for Langevin dynamics with decay rate ≈ γ (friction coefficient).
 *
 * @param vx_history Time series of x-velocities [time][particle]
 * @param vy_history Time series of y-velocities [time][particle]
 * @param max_lag Maximum time lag to compute (in number of snapshots)
 * @return Vector of VACF values for lags 0, 1, 2, ..., max_lag-1
 */
inline std::vector<float> compute_vacf(
    const std::vector<std::vector<float>>& vx_history,
    const std::vector<std::vector<float>>& vy_history,
    int max_lag) {
    
    if (vx_history.empty() || vy_history.empty()) {
        return std::vector<float>(max_lag, 0.0f);
    }
    
    size_t num_snapshots = vx_history.size();
    size_t N = vx_history[0].size();  // Number of particles
    
    // Limit max_lag to available data
    if (max_lag > static_cast<int>(num_snapshots)) {
        max_lag = static_cast<int>(num_snapshots);
    }
    
    // Initialize VACF and count arrays
    std::vector<float> vacf(max_lag, 0.0f);
    std::vector<int> counts(max_lag, 0);
    
    // Step 1: Compute ⟨v²⟩ for normalization
    float v_squared_avg = 0.0f;
    for (size_t t = 0; t < num_snapshots; ++t) {
        for (size_t i = 0; i < N; ++i) {
            v_squared_avg += vx_history[t][i] * vx_history[t][i] +
                           vy_history[t][i] * vy_history[t][i];
        }
    }
    v_squared_avg /= (num_snapshots * N);
    
    // Handle edge case: if all velocities are zero
    if (v_squared_avg < 1e-10f) {
        return vacf;  // Return all zeros
    }
    
    // Step 2: Compute correlation ⟨v(t) · v(t+τ)⟩ for each lag
    for (size_t t_ref = 0; t_ref < num_snapshots; ++t_ref) {
        for (int lag = 0; lag < max_lag && t_ref + lag < num_snapshots; ++lag) {
            size_t t_lag = t_ref + lag;
            
            // Accumulate dot product over all particles
            for (size_t i = 0; i < N; ++i) {
                float dot_product = vx_history[t_ref][i] * vx_history[t_lag][i] +
                                  vy_history[t_ref][i] * vy_history[t_lag][i];
                vacf[lag] += dot_product;
            }
            counts[lag]++;
        }
    }
    
    // Step 3: Normalize by count and ⟨v²⟩
    for (int lag = 0; lag < max_lag; ++lag) {
        if (counts[lag] > 0) {
            vacf[lag] /= (counts[lag] * N);  // Average over reference times AND particles
            vacf[lag] /= v_squared_avg;       // Normalize so VACF(0) = 1
        }
    }
    
    return vacf;
}

/**
 * @brief Save VACF data to CSV file for analysis
 *
 * Creates a CSV file with columns: lag,vacf
 *
 * @param vacf_values Vector of VACF values
 * @param dt Timestep size (to convert lag index to time)
 * @param measurement_interval Steps between velocity recordings
 * @param filename Output filename (e.g., "output/vacf_data.csv")
 * @return true if successful, false if file write failed
 */
inline bool save_vacf_to_csv(const std::vector<float>& vacf_values,
                             float dt,
                             int measurement_interval,
                             const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << "\n";
        return false;
    }
    
    // Write header
    file << "lag,vacf\n";
    
    // Write data (lag in time units)
    for (size_t i = 0; i < vacf_values.size(); ++i) {
        float lag_time = i * dt * measurement_interval;
        file << lag_time << "," << vacf_values[i] << "\n";
    }
    
    file.close();
    return true;
}

#endif // ANALYSIS_HPP