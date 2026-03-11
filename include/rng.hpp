#ifndef RNG_HPP
#define RNG_HPP

#include <cmath>
#include <cstdint>
#include <limits>

// M_PI is not standard in C++, so define it if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/**
 * PCG random number generator (XSH-RR variant)
 *
 * Box-Muller spare value is stored as a MEMBER (has_spare / spare) so that:
 *   - Every PCG instance is independent → no shared static state
 *   - The compiler can see no cross-iteration dependency → auto-vectorisation
 *   - Multiple threads each hold their own PCG → trivially thread-safe
 */
class PCG {
    private:
        uint64_t state;
        uint64_t inc;

        // Box-Muller pair cache — one spare Gaussian per RNG instance
        bool  has_spare = false;
        float spare     = 0.0f;

    public:
        // Constructor: Initializes the state and the stream (inc)
        explicit PCG(uint64_t seed) {
            state = 0U;

            // Since the contract does not allow passing a stream ID,
            // we default 'inc' to a fixed constant.
            // It MUST be odd for the LCG to have a full period.
            inc = (1442695040888963407ULL << 1u) | 1u;

            // "Prime" the state (standard PCG initialisation dance)
            // 1. Update state
            state = state * 6364136223846793005ULL + inc;
            // 2. Add seed
            state += seed;
            // 3. Update state again to mix the seed in
            state = state * 6364136223846793005ULL + inc;
        }

        uint32_t next() {
            // 1. Save old state for output calculation
            uint64_t old_state = state;

            // 2. Advance internal state
            state = old_state * 6364136223846793005ULL + inc;

            // 3. Calculate Output (XSH-RR)
            uint32_t xorshifted = static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
            uint32_t rot        = old_state >> 59u;

            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

        float uniform() {
            // Returns a float in range [0, 1)
            return next() * 2.3283064365386963e-10f;
        }

        /**
         * Generate Gaussian-distributed random numbers using Box-Muller transform.
         * Caches the second value in *this* (no static state) for 2× efficiency.
         *
         * Thread-safe: every PCG instance owns its own spare slot.
         * Vectorisable: the compiler sees no cross-call global dependency.
         *
         * @param mean   Mean of the Gaussian distribution (default 0)
         * @param stddev Standard deviation (default 1)
         * @return A sample from N(mean, stddev²)
         */
        float gaussian(float mean = 0.0f, float stddev = 1.0f) {
            if (has_spare) {
                has_spare = false;
                return spare * stddev + mean;
            }

            // Generate new pair using Box-Muller transform
            float u1;
            do {
                u1 = uniform();
            } while (u1 <= std::numeric_limits<float>::min());  // Avoid log(0)

            float u2 = uniform();

            // Box-Muller: generates TWO independent Gaussian samples
            float mag = std::sqrt(-2.0f * std::log(u1));
            float z0  = mag * std::cos(2.0f * static_cast<float>(M_PI) * u2);
            float z1  = mag * std::sin(2.0f * static_cast<float>(M_PI) * u2);

            // Cache z1 for next call
            spare     = z1;
            has_spare = true;

            return z0 * stddev + mean;
        }
};

/**
 * Free-function wrapper kept for backward compatibility.
 * Delegates to PCG::gaussian() — no static state.
 */
inline float gaussian_random(PCG& rng, float mean = 0.0f, float stddev = 1.0f) {
    return rng.gaussian(mean, stddev);
}

#endif // RNG_HPP
