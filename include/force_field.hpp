#ifndef FORCE_FIELD_HPP
#define FORCE_FIELD_HPP

#include <utility>  // For std::pair

/**
 * Compute force at position (x, y)
 * Returns: {Fx, Fy} as std::pair<float, float>
 *
 * Option A: Zero force (pure Brownian motion - simplest!)
 * Option B: Harmonic potential (particles attracted to center)
 */
inline std::pair<float, float> compute_force(float x, float y) {
    // OPTION A: Pure diffusion (no external forces)
    (void) x;
    (void) y;
    return {0.0f, 0.0f};
    
    // OPTION B: Harmonic confinement (uncomment to enable)
    // constexpr float k_spring = 0.01f;  // Weak spring for gentle confinement
    // constexpr float center = 50.0f;    // center of domain
    // return {-k_spring * (x - center), -k_spring * (y - center)};
}

#endif