#ifndef POTENTIAL_FIELD_HPP
#define POTENTIAL_FIELD_HPP

#include <vector>
#include <cmath>
#include <utility>
#include <stdexcept>

/**
 * Parametric potential field using Radial Basis Functions (RBF)
 *
 * Physics: U(x,y) = Σᵢ Aᵢ · exp(-|x-μᵢ|² / 2σᵢ²)
 * Force: F(x,y) = -∇U = Σᵢ Aᵢ · [(x-μₓ)/σ², (y-μᵧ)/σ²] · φ(x,y | μᵢ, σᵢ)
 *
 * Optimisations applied:
 *   - inv_2sigma2[i] = 1/(2σᵢ²) precomputed once from widths (never change mid-episode)
 *   - inv_sigma2[i]  = 1/σᵢ²    precomputed for the gradient coefficient
 *   Both are recomputed only when set_parameters() replaces widths.
 */
class PotentialField {
private:
    // Parameters for each basis function
    std::vector<float> centers_x;    // μₓ for each basis
    std::vector<float> centers_y;    // μᵧ for each basis
    std::vector<float> strengths;    // Aᵢ (amplitude)
    std::vector<float> widths;       // σᵢ (spatial scale)

    // Precomputed reciprocals — recomputed whenever widths change
    std::vector<float> inv_2sigma2;  // 1 / (2 σᵢ²)  — exponent denominator
    std::vector<float> inv_sigma2;   // 1 / σᵢ²       — gradient coefficient

    size_t num_basis;               // Number of basis functions

    /** Recompute the cached reciprocals from the current widths array. */
    void recompute_sigma_cache() {
        inv_2sigma2.resize(num_basis);
        inv_sigma2.resize(num_basis);
        for (size_t i = 0; i < num_basis; ++i) {
            float s2        = widths[i] * widths[i];
            inv_sigma2[i]   = 1.0f / s2;
            inv_2sigma2[i]  = 0.5f * inv_sigma2[i];  // 1/(2σ²)
        }
    }

public:
    /**
     * Constructor: Initialise with given number of basis functions.
     * Default: uniformly distributed centres on a grid, zero strength.
     *
     * @param n_basis     Number of RBF basis functions
     * @param domain_size Physical size of simulation domain
     */
    PotentialField(size_t n_basis, float domain_size)
        : num_basis(n_basis)
    {
        centers_x.resize(n_basis);
        centers_y.resize(n_basis);
        strengths.resize(n_basis, 0.0f);
        widths.resize(n_basis, domain_size / 10.0f);

        // Initialise centres on a regular grid
        int   grid_size = static_cast<int>(std::ceil(std::sqrt(static_cast<float>(n_basis))));
        float spacing   = domain_size / grid_size;

        for (size_t i = 0; i < n_basis; ++i) {
            centers_x[i] = (i % grid_size) * spacing + spacing / 2.0f;
            centers_y[i] = (i / grid_size) * spacing + spacing / 2.0f;
        }

        // Precompute reciprocals from initial widths
        recompute_sigma_cache();
    }

    /**
     * Compute force at position (x, y).
     *
     * Uses precomputed inv_2sigma2 / inv_sigma2 — avoids per-call divisions
     * and repeated widths[i]*widths[i] inside the hot N-particle loop.
     *
     * @param x X-coordinate
     * @param y Y-coordinate
     * @return {Fx, Fy} Force vector at position
     */
    std::pair<float, float> compute_force(float x, float y) const {
        float Fx = 0.0f;
        float Fy = 0.0f;

        for (size_t i = 0; i < num_basis; ++i) {
            float dx  = x - centers_x[i];
            float dy  = y - centers_y[i];
            float r2  = dx * dx + dy * dy;

            // Gaussian basis: φ = exp(-r²·inv_2sigma2)  (no division in exponent)
            float phi        = std::exp(-r2 * inv_2sigma2[i]);

            // Gradient coefficient: A · φ / σ²  (inv_sigma2 already precomputed)
            float grad_coeff = strengths[i] * phi * inv_sigma2[i];

            Fx += grad_coeff * dx;
            Fy += grad_coeff * dy;
        }

        return {Fx, Fy};
    }

    /**
     * Set strengths (called by RL agent as action).
     * Widths do NOT change → sigma cache remains valid.
     *
     * @param new_strengths Array of amplitudes for each basis function
     */
    void set_strengths(const std::vector<float>& new_strengths) {
        if (new_strengths.size() != num_basis) {
            throw std::invalid_argument("Strength array size mismatch");
        }
        strengths = new_strengths;
    }

    /**
     * Set all parameters at once (advanced control).
     * Widths may change → sigma cache is recomputed.
     *
     * @param cx  Centre x-coordinates
     * @param cy  Centre y-coordinates
     * @param amp Amplitudes / strengths
     * @param sig Widths / sigmas
     */
    void set_parameters(const std::vector<float>& cx,
                        const std::vector<float>& cy,
                        const std::vector<float>& amp,
                        const std::vector<float>& sig) {
        if (cx.size()  != num_basis || cy.size()  != num_basis ||
            amp.size() != num_basis || sig.size() != num_basis) {
            throw std::invalid_argument("Parameter array size mismatch");
        }
        centers_x = cx;
        centers_y = cy;
        strengths = amp;
        widths    = sig;

        // Widths changed — must recompute cached reciprocals
        recompute_sigma_cache();
    }

    // Getters for inspection / visualisation
    size_t get_num_basis()                       const { return num_basis;  }
    const std::vector<float>& get_centers_x()   const { return centers_x; }
    const std::vector<float>& get_centers_y()   const { return centers_y; }
    const std::vector<float>& get_strengths()   const { return strengths;  }
    const std::vector<float>& get_widths()      const { return widths;     }
};

#endif // POTENTIAL_FIELD_HPP
