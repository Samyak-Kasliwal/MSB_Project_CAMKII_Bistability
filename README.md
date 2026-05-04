# CaMKII Gamma/M Coupling Pipeline

This project runs stochastic CaMKII switch simulations. The main analysis is
`my_model`, which follows the attached notebook
`C:\Users\Samyak\Downloads\camkii_simulation.ipynb` for the core stochastic
model.

The most important question is:

```text
How does CaMKII memory lifetime depend on the coupling gamma and the number
of holoenzymes M when exchange is normalized as gamma/M?
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the default analysis:

```powershell
python main.py
```

The default in `main.py` is:

```python
config.runs.selected_runs = ["my_model"]
```

Because `my_model` depends on `simple_sanity_check`, the pipeline will run:

```text
simple_sanity_check
my_model
```

Use this to run everything:

```python
config.runs.selected_runs = ["all"]
```

## Run List

The pipeline has exactly these run ids, in this order:

1. `simple_sanity_check`
2. `zabotinsky_reproduction`
3. `miller_reproduction`
4. `my_model`
5. `caliberation_checks`
6. `modified_autophosphrylation_propensity_check`
7. `calcium_spike_condition_analysis`

The spellings `caliberation_checks` and
`modified_autophosphrylation_propensity_check` match the requested names.
The corrected spellings are accepted as aliases in `config.py`.

## State Variables

A synapse contains `M` CaMKII holoenzymes.

Each holoenzyme contains:

```text
N = 12 subunits
```

Each subunit is either:

```text
OFF = unphosphorylated
ON  = phosphorylated
```

The state is the vector:

```text
n = (n_1, n_2, ..., n_M)
```

where:

```text
n_alpha = number of ON subunits in holoenzyme alpha
0 <= n_alpha <= N
```

The total number of ON subunits is:

```text
n_tot = sum_alpha n_alpha
```

The UP threshold per holoenzyme is:

```text
theta = 6 = N / 2
```

A holoenzyme is classified as UP if:

```text
n_alpha >= theta
```

The synapse-level fraction UP is:

```text
f_up = number of holoenzymes with n_alpha >= theta divided by M
```

The synapse is classified as UP if:

```text
f_up >= 0.5
```

This matches the notebook's lifetime rule.

## Notebook-Exact My Model

The `my_model` run uses four stochastic reaction channels.

### R1: Autophosphorylation

Within holoenzyme `alpha`, an OFF subunit becomes ON:

```text
n_alpha -> n_alpha + 1
```

The propensity is:

```text
a1_alpha = r1 * (N - n_alpha) * n_alpha / N
```

Interpretation:

```text
N - n_alpha = number of OFF target subunits
n_alpha / N = fraction of ON scaffold/catalyst subunits
r1          = autophosphorylation rate scale
```

Important detail:

```text
There is no epsilon basal term in my_model.
```

So if `n_alpha = 0`, then:

```text
a1_alpha = 0
```

This is exactly what the notebook implements.

### R2: PP1 Dephosphorylation

Within holoenzyme `alpha`, an ON subunit becomes OFF:

```text
n_alpha -> n_alpha - 1
```

The propensity is:

```text
a2_alpha = Vmax * n_alpha / (Km + n_tot)
```

The total R2 propensity is:

```text
A2 = sum_alpha a2_alpha
   = Vmax * n_tot / (Km + n_tot)
```

Interpretation:

```text
Vmax = maximum PP1 dephosphorylation rate
Km   = effective Michaelis constant in subunit-count units
```

Because the denominator uses `n_tot`, PP1 saturation is shared across the
whole synaptic CaMKII pool.

### R3: Subunit Exchange

An ON subunit migrates from holoenzyme `alpha` to holoenzyme `beta`:

```text
n_alpha -> n_alpha - 1
n_beta  -> n_beta + 1
```

for `alpha != beta`.

The propensity is:

```text
a3_alpha_beta = (gamma / M) * n_alpha * (N - n_beta)
```

Interpretation:

```text
n_alpha     = available ON donor subunits in alpha
N - n_beta  = available OFF acceptor slots in beta
gamma / M   = all-to-all coupling normalized by M
```

The total exchange propensity can be computed efficiently as:

```text
A3 = (gamma / M) *
     [ n_tot * (M*N - n_tot)
       - sum_alpha n_alpha * (N - n_alpha) ]
```

The subtraction removes invalid self-exchange terms where `alpha = beta`.

This `gamma / M` scaling is the core of the main analysis.

### R4: Fair-Sampling ON-Subunit Turnover

An ON subunit is degraded and replaced by a fresh OFF subunit:

```text
n_alpha -> n_alpha - 1
```

The notebook propensity is:

```text
a4_alpha = delta * n_alpha * n_tot / (M * N)
```

The total turnover propensity is:

```text
A4 = sum_alpha a4_alpha
   = delta * n_tot^2 / (M * N)
```

Interpretation:

```text
delta          = turnover rate scale
n_alpha        = ON subunits available in holoenzyme alpha
n_tot / (M*N)  = fair-sampling probability that a randomly sampled subunit is ON
```

This is different from a simple linear ON-subunit turnover term
`delta * n_alpha`. The notebook uses the fair-sampling quadratic form.

## Gillespie SSA

For a current state `n`, the total propensity is:

```text
A0 = A1 + A2 + A3 + A4
```

where:

```text
A1 = sum_alpha a1_alpha
A2 = sum_alpha a2_alpha
A3 = sum_{alpha != beta} a3_alpha_beta
A4 = sum_alpha a4_alpha
```

The waiting time to the next event is sampled as:

```text
tau = -log(u) / A0
```

where:

```text
u ~ Uniform(0, 1)
```

The event channel is selected with probability proportional to its propensity:

```text
P(event i) = a_i / A0
```

Then the selected state update is applied, time is advanced by `tau`, and the
process repeats until either:

```text
t >= t_max
```

or the requested transition is observed.

## Lifetime Definition

The main lifetime is the UP-to-DOWN lifetime.

The initial state for `my_model` is:

```text
n_alpha = N for all alpha
```

So:

```text
n_tot = M*N
f_up = 1
```

An UP-to-DOWN transition is recorded when:

```text
previous f_up >= 0.5
current  f_up <  0.5
```

The observed lifetime is the transition time.

If no transition occurs before `t_max`, the trajectory is censored:

```text
observed_time = t_max
censored = 1
transition_found = 0
```

If a transition occurs:

```text
observed_time = transition_time
censored = 0
transition_found = 1
```

## Censored Exponential Lifetime Estimate

For each condition, the code pools trajectories.

Let:

```text
t_i = observed time for trajectory i
d_i = 1 if transition observed, 0 if censored
```

The censored exponential maximum-likelihood estimate is:

```text
lambda_hat = sum_i d_i / sum_i t_i
tau_hat    = 1 / lambda_hat
           = sum_i t_i / sum_i d_i
```

If no transitions are observed:

```text
lambda_hat = 0
tau_hat = infinity
```

Bootstrap confidence intervals are computed by resampling trajectories within
the condition.

## Gamma/M Coupling Analysis

The `my_model` run scans:

```text
M in config.simulation.my_model_m_scan
gamma in config.simulation.my_model_gamma_scan
```

Default values in `main.py`:

```text
M scan     = [2, 4, 6, 8]
gamma scan = [0.0, 0.001, 0.01, 0.1]
delta      = 2.0e-3 s^-1
t_max      = 1.0e5 s
```

For each gamma, the code fits:

```text
log(tau_UP) = slope(gamma) * M + intercept(gamma)
```

This tests the theoretical exponential scaling:

```text
tau_UP(M, gamma) approximately C(gamma) * exp[M * DeltaPhi(gamma)]
```

or equivalently:

```text
log tau_UP approximately M * DeltaPhi(gamma) + constant
```

The main outputs are:

```text
results/my_model/tables/my_model__gamma_M_coupling_summary.csv
results/my_model/tables/my_model__gamma_M_coupling_slopes.csv
```

## Simple Sanity Check

Run id:

```text
simple_sanity_check
```

This writes a half-phosphorylated propensity table using:

```text
n_alpha = theta for all alpha
```

It reports:

```text
channels = 3*M + M*(M - 1)
```

because the notebook model has:

```text
M R1 channels
M R2 channels
M R4 channels
M*(M - 1) R3 exchange channels
```

It also writes one short notebook-exact trajectory from the UP state.

Outputs:

```text
results/simple_sanity_check/tables/simple_sanity_check__propensities.csv
results/simple_sanity_check/raw_trajectories/simple_sanity_check__trace.csv
```

## Zabotinsky Reproduction

Run id:

```text
zabotinsky_reproduction
```

This implements the deterministic single-holoenzyme bistability analysis from
the notebook. The paper spelling is Zhabotinsky, but the run id preserves the
requested spelling.

For `M = 1`, the continuous approximation is:

```text
dn/dt = r1 * (N - n) * n / N
        - Vmax * n / (Km + n)
        - delta * n^2 / N
```

For `n > 0`, setting `dn/dt = 0` and dividing by `n` gives:

```text
r1 * (N - n) / N
- Vmax / (Km + n)
- delta * n / N = 0
```

The notebook rearranges this into the quadratic:

```text
(r1 + delta) * n^2
- [r1*N - Km*(r1 + delta)] * n
+ N*Vmax = 0
```

Define:

```text
A = r1 + delta
B = -[r1*N - Km*(r1 + delta)]
C = N*Vmax
```

Then:

```text
discriminant = B^2 - 4*A*C
```

Bistability requires:

```text
discriminant > 0
```

and both fixed points must lie in:

```text
0 < n < N
```

The fixed points are:

```text
n = (-B +/- sqrt(discriminant)) / (2*A)
```

The code scans delta and stores the lower and upper fixed points.

Outputs:

```text
results/zabotinsky_reproduction/tables/zabotinsky_reproduction__fixed_points.csv
results/zabotinsky_reproduction/tables/zabotinsky_reproduction__delta_scan.csv
```

## Miller Reproduction

Run id:

```text
miller_reproduction
```

This run keeps the older MZ-full stochastic model separate from the notebook
`my_model`.

The MZ-full model uses Ca-dependent first and later autophosphorylation terms:

```text
hill_1 = Ca^h / (KH1^h + Ca^h)
hill_2 = Ca^h / (KH2^h + Ca^h)
```

Initial phosphorylation from `n_alpha = 0`:

```text
rf1_alpha = k1 * N * hill_1
```

Later phosphorylation for `0 < n_alpha < N`:

```text
rf2_alpha = k2 * (N - n_alpha) * (n_alpha / N) * hill_2
```

Dephosphorylation:

```text
dephospho_alpha = k_cat * n_alpha / (Km + n_tot)
```

Holoenzyme turnover:

```text
turnover_alpha = delta_holo
```

A holoenzyme turnover event resets:

```text
n_alpha -> 0
```

This reproduction scans `M` and fits:

```text
log(tau_UP) = slope * M + intercept
```

Outputs:

```text
results/miller_reproduction/tables/miller_reproduction__condition_summary.csv
results/miller_reproduction/tables/miller_reproduction__slope_fit.csv
```

## Caliberation Checks

Run id:

```text
caliberation_checks
```

This run uses `my_model` and sweeps one parameter at a time while holding the
others fixed.

The swept parameters are:

```text
r1
gamma
delta
Vmax
```

The default sweep values are set in `main.py`:

```text
r1    = [1.0, 1.5, 2.0]
gamma = config.simulation.my_model_gamma_scan
delta = [0.0, 1.0e-5, 1.0e-4, 1.0e-3, 2.0e-3]
Vmax  = [1.0, 2.0, 4.0]
```

This checks how sensitive the gamma/M result is to the model's key rate
parameters.

Output:

```text
results/caliberation_checks/tables/caliberation_checks__summary.csv
```

## Modified Autophosphrylation Propensity Check

Run id:

```text
modified_autophosphrylation_propensity_check
```

This is an addition on top of `my_model`.

The baseline R1 event is:

```text
n_alpha -> n_alpha + 1
```

The modified event is:

```text
n_alpha -> n_alpha + 2
```

with clipping at `N`:

```text
n_alpha_new = min(n_alpha + 2, N)
```

The propensity is not doubled. The event propensity remains:

```text
a1_alpha = r1 * (N - n_alpha) * n_alpha / N
```

Only the state update changes. The run compares:

```text
autophosphorylation_step = 1
autophosphorylation_step = 2
```

across the same `M` and `gamma` grid as `my_model`.

Outputs:

```text
results/modified_autophosphrylation_propensity_check/tables/modified_autophosphrylation_propensity_check__summary.csv
results/modified_autophosphrylation_propensity_check/tables/modified_autophosphrylation_propensity_check__slopes.csv
```

## Calcium Spike Condition Analysis

Run id:

```text
calcium_spike_condition_analysis
```

This is another addition on top of `my_model`.

The baseline model uses constant:

```text
r1
```

During calcium-spike windows, the code temporarily uses:

```text
r1_spike = r1 * calcium_spike_r1_multiplier
```

The default spike protocol in `main.py` is:

```text
first spike start = 500 s
spike duration    = 20 s
spike period      = 1000 s
spike count       = 20
r1 multiplier     = 8
```

So spike `j` is active during:

```text
start_j <= t < start_j + duration
```

where:

```text
start_j = calcium_spike_start_seconds
          + j * calcium_spike_period_seconds
```

for:

```text
j = 0, 1, ..., calcium_spike_count - 1
```

During a spike, only the R1 propensity changes:

```text
a1_alpha_spike = (r1 * multiplier) * (N - n_alpha) * n_alpha / N
```

R2, R3, and R4 remain unchanged.

The run compares:

```text
calcium_spike_enabled = False
calcium_spike_enabled = True
```

across the same `M` and `gamma` grid as `my_model`.

Outputs:

```text
results/calcium_spike_condition_analysis/tables/calcium_spike_condition_analysis__summary.csv
results/calcium_spike_condition_analysis/tables/calcium_spike_condition_analysis__slopes.csv
```

## Output Columns

Condition summary tables include:

```text
run_id
model
transition
M
N
theta
r1
Vmax
Km
gamma
delta
autophosphorylation_step
calcium_spike_enabled
n_trajectories
t_max
valid_trajectories
invalid_trajectories
uncensored_events
censored_observations
reported_lifetime
reported_lifetime_lower
reported_lifetime_upper
tau_hat_mle
kaplan_meier_median
survival_r_squared
mean_final_f_up
```

Raw trajectory CSV files include one row per trajectory with:

```text
observed_time
transition_found
censored
final_n_tot
final_f_up
event_count
final_state
```

`final_state` is stored as a semicolon-separated vector:

```text
n_1;n_2;...;n_M
```

## Checkpoints And Resume

Every condition writes partial results in:

```text
checkpoints/<run_id>/
```

Completed raw trajectories are written to:

```text
results/<run_id>/raw_trajectories/
```

The pipeline state is stored in:

```text
checkpoints/pipeline_state.json
```

By default, completed work is reused:

```python
config.runs.skip_completed_runs = True
```

To rebuild selected runs:

```python
config.runs.force_recompute_selected_runs = True
```

## Plots

Plots are off by default for speed:

```python
config.output.generate_plots = False
```

Enable them with:

```python
config.output.generate_plots = True
```

Plot files are written under:

```text
results/<run_id>/plots/
```

## Main Configuration Values

The most important settings are in `main.py`:

```python
config.simulation.my_model_m_scan
config.simulation.my_model_gamma_scan
config.simulation.my_model_delta
config.simulation.my_model_t_max
config.runs.my_model_trajectories
config.runs.bootstrap_replicates
```

Increase trajectory counts for final estimates. The defaults are intentionally
small enough to make the pipeline runnable while developing.
