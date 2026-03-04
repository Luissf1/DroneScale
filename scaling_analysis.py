"""
Scaling Analysis of PSO-Optimized PID Controllers for Quadrotors of Different Sizes
Trajectories 2026 Short Paper

Author: Luis Adrián Silva Reyes
Based on thesis work: "Optimización bio-inspirada de controladores PID para mejora 
del desempeño en vehículos aéreos no tripulados"

This code performs:
1. PSO optimization for quadrotors of different sizes (0.5x, 1x, 2x, 5x mass)
2. Derivation of scaling laws from optimal gains
3. Validation of gain transfer with and without scaling
4. Generation of all figures and tables for the paper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd
import os
import warnings
from datetime import datetime
import json
from tqdm import tqdm
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class QuadrotorScalingAnalysis:
    """
    Complete analysis of scaling laws for PSO-optimized PID controllers
    across quadrotors of different sizes.
    """
    
    def __init__(self, modo_rapido=False):
        """Initialize with configuration parameters"""
        self.modo_rapido = modo_rapido
        
        # Baseline parameters (1x)
        self.baseline_params = {
            'm': 1.0,           # kg
            'g': 9.81,          # m/s²
            'l': 0.25,          # m
            'Ixx': 0.1,         # kg·m²
            'Iyy': 0.1,         # kg·m²
            'Izz': 0.2,         # kg·m²
            'b': 3.13e-5,       # N·s²
            'd': 7.5e-7,        # N·m·s²
            'Kf': 0.1           # Fricción
        }
        
        # Define size variants (scaling factor for mass)
        self.size_factors = [0.5, 1.0, 2.0, 5.0]
        self.size_names = ['Micro (0.5×)', 'Baseline (1×)', 'Medium (2×)', 'Large (5×)']
        self.size_codes = ['M0.5', 'M1.0', 'M2.0', 'M5.0']
        
        # Generate parameters for each size (assuming geometric similarity)
        self.variants = {}
        for factor, name in zip(self.size_factors, self.size_names):
            # Mass scales linearly with factor
            m = self.baseline_params['m'] * factor
            
            # Arm length scales as cube root of mass (assuming constant density)
            l = self.baseline_params['l'] * (factor ** (1/3))
            
            # Moments of inertia scale as m * l²
            I_scale = factor * (factor ** (2/3))  # m * l² scaling
            
            self.variants[factor] = {
                'name': name,
                'code': f'M{factor}',
                'params': {
                    'm': m,
                    'l': l,
                    'Ixx': self.baseline_params['Ixx'] * I_scale,
                    'Iyy': self.baseline_params['Iyy'] * I_scale,
                    'Izz': self.baseline_params['Izz'] * I_scale,
                    'g': self.baseline_params['g'],
                    'b': self.baseline_params['b'],
                    'd': self.baseline_params['d'],
                    'Kf': self.baseline_params['Kf']
                }
            }
        
        # PSO parameters (from thesis)
        if self.modo_rapido:
            self.nPop = 10
            self.MaxIter = 10
            self.num_ejecuciones = 3
        else:
            self.nPop = 50
            self.MaxIter = 100
            self.num_ejecuciones = 30
        
        self.nVar = 12
        self.w_max = 0.7
        self.w_min = 0.4
        self.c1 = 1.7
        self.c2 = 1.7
        self.vel_max = 0.2
        
        # Search limits (from thesis)
        self.VarMin = np.array([
            2.0, 0.01, 0.1,     # Altitude (Z)
            0.1, 0.001, 0.1,    # Roll (φ)
            0.1, 0.001, 0.1,    # Pitch (θ)
            0.1, 0.001, 0.1     # Yaw (ψ)
        ])
        
        self.VarMax = np.array([
            15.0, 2.0, 5.0,     # Altitude (Z)
            10.0, 0.1, 2.0,     # Roll (φ)
            10.0, 0.1, 2.0,     # Pitch (θ)
            10.0, 0.1, 2.0      # Yaw (ψ)
        ])
        
        # Cost function weights
        self.pesos = [0.3, 0.3, 0.2, 0.2]  # ts, Mp, ITSE, IAE
        
        # Simulation settings
        self.t_simulacion = (0, 10)
        self.n_puntos = 500
        self.tol_settling = 0.02
        
        # Flight scenarios (from thesis)
        self.escenarios = np.array([
            [1.0, 0.0, 0.0, 0.0],                 # E1: Stationary takeoff
            [1.5, 0.1, -0.1, 0.0],                # E2: Inclined takeoff
            [2.0, -0.2, 0.2, 0.0],                # E3: Transitional climb
            [1.0, 0.0, 0.0, np.pi / 4],           # E4: Yaw-controlled takeoff
            [0.5, -0.1, -0.1, -np.pi / 6]         # E5: Multi-axis maneuver
        ])
        
        self.nombres_escenarios = [
            "Stationary Takeoff",
            "Inclined Takeoff",
            "Transitional Climb",
            "Yaw-Controlled Takeoff",
            "Multi-Axis Maneuver"
        ]
        
        # Ziegler-Nichols baseline values (from thesis)
        self.zn_gains = np.array([
            12.50, 1.250, 3.125,   # Altitude
            6.250, 0.075, 1.250,   # Roll
            6.250, 0.075, 1.250,   # Pitch
            5.000, 0.050, 1.000    # Yaw
        ])
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = f"Scaling_Analysis_{timestamp}"
        self.fig_dir = os.path.join(self.out_dir, "figures")
        self.tab_dir = os.path.join(self.out_dir, "tables")
        self.data_dir = os.path.join(self.out_dir, "data")
        
        for d in [self.out_dir, self.fig_dir, self.tab_dir, self.data_dir]:
            os.makedirs(d, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("QUADROTOR PID SCALING ANALYSIS")
        print(f"{'='*60}")
        print(f"Output directory: {self.out_dir}")
        print(f"Size variants: {', '.join(self.size_names)}")
        print(f"Mode: {'RAPID (test)' if self.modo_rapido else 'FULL'}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # QUADROTOR DYNAMICS MODEL
    # =========================================================================
    
    def modelo_cuadrotor(self, t, X, ganancias, ref, params):
        """Quadrotor dynamics model with given parameters"""
        x, y, z, phi, theta, psi, vx, vy, vz, p, q, r = X
        z_des, phi_des, theta_des, psi_des = ref
        
        # Extract gains
        Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi, Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi = ganancias
        
        # Errors
        e_z = z_des - z
        e_phi = phi_des - phi
        e_theta = theta_des - theta
        e_psi = psi_des - psi
        
        # Error derivatives
        de_z = -vz
        de_phi = -p
        de_theta = -q
        de_psi = -r
        
        # Integral terms (approximation)
        int_e_z = e_z * t if t > 0 else 0
        int_e_phi = e_phi * t if t > 0 else 0
        int_e_theta = e_theta * t if t > 0 else 0
        int_e_psi = e_psi * t if t > 0 else 0
        
        # PID control law
        U1 = Kp_z * e_z + Ki_z * int_e_z + Kd_z * de_z
        U2 = Kp_phi * e_phi + Ki_phi * int_e_phi + Kd_phi * de_phi
        U3 = Kp_theta * e_theta + Ki_theta * int_e_theta + Kd_theta * de_theta
        U4 = Kp_psi * e_psi + Ki_psi * int_e_psi + Kd_psi * de_psi
        
        # Saturation
        U_max = params['m'] * params['g'] * 2.0
        U1 = np.clip(U1, 0, U_max)
        U2 = np.clip(U2, -3, 3)
        U3 = np.clip(U3, -3, 3)
        U4 = np.clip(U4, -1, 1)
        
        # Trigonometry
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        
        # Forces in body frame
        Fx = (spsi * sphi + cpsi * stheta * cphi) * U1
        Fy = (-cpsi * sphi + spsi * stheta * cphi) * U1
        Fz = (ctheta * cphi) * U1
        
        m = params['m']
        g = params['g']
        Kf = params['Kf']
        
        # Linear accelerations
        ax = Fx / m - Kf * vx
        ay = Fy / m - Kf * vy
        az = Fz / m - g - Kf * vz
        
        # Angular accelerations
        Ixx, Iyy, Izz = params['Ixx'], params['Iyy'], params['Izz']
        l = params['l']
        
        p_dot = (l * U2 + q * r * (Iyy - Izz)) / Ixx
        q_dot = (l * U3 + p * r * (Izz - Ixx)) / Iyy
        r_dot = (U4 + p * q * (Ixx - Iyy)) / Izz
        
        return np.array([vx, vy, vz, p, q, r, ax, ay, az, p_dot, q_dot, r_dot])
    
    def evaluar_pid(self, ganancias, ref, params):
        """Evaluate PID performance for given gains and parameters"""
        try:
            X0 = np.zeros(12)
            t_eval = np.linspace(self.t_simulacion[0], self.t_simulacion[1], self.n_puntos)
            
            sol = solve_ivp(
                lambda t, X: self.modelo_cuadrotor(t, X, ganancias, ref, params),
                self.t_simulacion,
                X0,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-4,
                atol=1e-6
            )
            
            if not sol.success:
                return 1000.0, self._metricas_default()
            
            t = sol.t
            X = sol.y
            z = X[2, :]
            
            # Calculate metrics
            e_z = ref[0] - z
            RMSE = np.sqrt(np.mean(e_z ** 2))
            IAE = np.trapz(np.abs(e_z), t)
            ITSE = np.trapz(t * e_z ** 2, t)
            
            # Settling time
            tolerancia = self.tol_settling * abs(ref[0]) if ref[0] != 0 else 0.02
            dentro_tolerancia = np.abs(e_z) <= tolerancia
            
            t_settling = t[-1]
            if np.any(dentro_tolerancia):
                for i in range(len(t) - 1, -1, -1):
                    if i > 0 and np.all(dentro_tolerancia[i:]):
                        t_settling = t[i]
                        break
            
            # Overshoot
            max_z = np.max(z) if len(z) > 0 else 0
            sobrepico = max(0, (max_z - ref[0]) / ref[0] * 100) if ref[0] != 0 else 0
            
            metricas = {
                'RMSE': RMSE,
                'IAE': IAE,
                'ITSE': ITSE,
                't_settling': t_settling,
                'sobrepico': sobrepico
            }
            
            # Fitness (lower is better)
            t_norm = min(t_settling / 10.0, 1.0)
            mp_norm = min(sobrepico / 100.0, 1.0)
            itse_norm = min(ITSE / 50.0, 1.0)
            iae_norm = min(IAE / 20.0, 1.0)
            
            fitness = (self.pesos[0] * t_norm + 
                      self.pesos[1] * mp_norm + 
                      self.pesos[2] * itse_norm + 
                      self.pesos[3] * iae_norm)
            
            return fitness, metricas
            
        except Exception as e:
            return 1000.0, self._metricas_default()
    
    def _metricas_default(self):
        return {'RMSE': 10.0, 'IAE': 50.0, 'ITSE': 100.0, 
                't_settling': 10.0, 'sobrepico': 100.0}
    
    # =========================================================================
    # PSO ALGORITHM
    # =========================================================================
    
    def pso_optimize(self, params, ref_escenario=None, num_ejecuciones=1):
        """Run PSO optimization for given parameters"""
        if ref_escenario is None:
            ref_escenario = self.escenarios[0]  # Default to E1
        
        best_results = []
        
        for ejec in range(num_ejecuciones):
            np.random.seed(42 + ejec)
            
            # Initialize swarm
            particles = []
            gbest_pos = None
            gbest_fit = float('inf')
            
            for i in range(self.nPop):
                pos = np.random.uniform(self.VarMin, self.VarMax)
                vel = np.zeros(self.nVar)
                fit, _ = self.evaluar_pid(pos, ref_escenario, params)
                
                particles.append({
                    'pos': pos,
                    'vel': vel,
                    'fit': fit,
                    'pbest_pos': pos.copy(),
                    'pbest_fit': fit
                })
                
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = pos.copy()
            
            # Main loop
            convergence = []
            for iter in range(self.MaxIter):
                w = self.w_max - (self.w_max - self.w_min) * (iter / self.MaxIter)
                
                for i in range(self.nPop):
                    r1, r2 = np.random.rand(2)
                    
                    # Velocity update
                    cognitive = self.c1 * r1 * (particles[i]['pbest_pos'] - particles[i]['pos'])
                    social = self.c2 * r2 * (gbest_pos - particles[i]['pos'])
                    
                    particles[i]['vel'] = w * particles[i]['vel'] + cognitive + social
                    
                    # Velocity clamping
                    vel_max = self.vel_max * (self.VarMax - self.VarMin)
                    particles[i]['vel'] = np.clip(particles[i]['vel'], -vel_max, vel_max)
                    
                    # Position update
                    particles[i]['pos'] += particles[i]['vel']
                    particles[i]['pos'] = np.clip(particles[i]['pos'], self.VarMin, self.VarMax)
                    
                    # Evaluate
                    fit, _ = self.evaluar_pid(particles[i]['pos'], ref_escenario, params)
                    particles[i]['fit'] = fit
                    
                    # Update personal best
                    if fit < particles[i]['pbest_fit']:
                        particles[i]['pbest_pos'] = particles[i]['pos'].copy()
                        particles[i]['pbest_fit'] = fit
                        
                        # Update global best
                        if fit < gbest_fit:
                            gbest_fit = fit
                            gbest_pos = particles[i]['pos'].copy()
                
                convergence.append(gbest_fit)
            
            best_results.append({
                'pos': gbest_pos,
                'fit': gbest_fit,
                'convergence': convergence
            })
        
        # Return best overall
        best_idx = np.argmin([r['fit'] for r in best_results])
        return best_results[best_idx]
    
    # =========================================================================
    # SCALING ANALYSIS
    # =========================================================================
    
    def run_full_analysis(self):
        """Run complete scaling analysis"""
        print("\n1. OPTIMIZING BASELINE QUADROTOR (1×)")
        baseline_params = self.variants[1.0]['params']
        
        # Optimize for each scenario and average
        baseline_gains_scenarios = []
        
        for idx, (esc, esc_name) in enumerate(zip(self.escenarios, self.nombres_escenarios)):
            print(f"   Scenario {idx+1}: {esc_name}")
            result = self.pso_optimize(baseline_params, esc, num_ejecuciones=self.num_ejecuciones)
            baseline_gains_scenarios.append(result['pos'])
        
        # Average gains across scenarios
        baseline_gains = np.mean(baseline_gains_scenarios, axis=0)
        print(f"\n✅ Baseline gains obtained")
        
        # Store results
        results = {
            'baseline': {
                'factor': 1.0,
                'name': 'Baseline (1×)',
                'gains': baseline_gains,
                'params': baseline_params
            }
        }
        
        print("\n2. OPTIMIZING OTHER SIZE VARIANTS")
        optimal_gains = {1.0: baseline_gains}
        
        for factor in [0.5, 2.0, 5.0]:
            print(f"\n   {self.variants[factor]['name']} (factor={factor})")
            params = self.variants[factor]['params']
            
            # Optimize for this size
            gains_scenarios = []
            for idx, esc in enumerate(self.escenarios):
                result = self.pso_optimize(params, esc, num_ejecuciones=self.num_ejecuciones)
                gains_scenarios.append(result['pos'])
            
            optimal_gains[factor] = np.mean(gains_scenarios, axis=0)
            
            results[factor] = {
                'factor': factor,
                'name': self.variants[factor]['name'],
                'gains': optimal_gains[factor],
                'params': params
            }
        
        print("\n3. EVALUATING GAIN TRANSFER")
        transfer_results = self.evaluate_gain_transfer(optimal_gains)
        
        # Save results
        self.save_results(results, transfer_results)
        
        # Generate figures
        self.generate_figures(results, transfer_results)
        
        # Generate tables
        self.generate_tables(results, transfer_results)
        
        return results, transfer_results
    
    def evaluate_gain_transfer(self, optimal_gains):
        """Evaluate performance when transferring gains between sizes"""
        transfer_results = []
        
        # For each target size
        for target_factor in [0.5, 1.0, 2.0, 5.0]:
            target_params = self.variants[target_factor]['params']
            target_optimal = optimal_gains[target_factor]
            
            # Evaluate optimal (baseline) performance
            fit_optimal_sum = 0
            for esc in self.escenarios:
                fit, _ = self.evaluar_pid(target_optimal, esc, target_params)
                fit_optimal_sum += fit
            fit_optimal = fit_optimal_sum / len(self.escenarios)
            
            # Evaluate Ziegler-Nichols performance
            fit_zn_sum = 0
            for esc in self.escenarios:
                fit, _ = self.evaluar_pid(self.zn_gains, esc, target_params)
                fit_zn_sum += fit
            fit_zn = fit_zn_sum / len(self.escenarios)
            
            # For each source size
            for source_factor in [0.5, 1.0, 2.0, 5.0]:
                if source_factor == target_factor:
                    continue
                
                source_gains = optimal_gains[source_factor]
                source_params = self.variants[source_factor]['params']
                
                # Unscaled transfer
                fit_unscaled_sum = 0
                for esc in self.escenarios:
                    fit, _ = self.evaluar_pid(source_gains, esc, target_params)
                    fit_unscaled_sum += fit
                fit_unscaled = fit_unscaled_sum / len(self.escenarios)
                
                # Scaled transfer using derived laws
                scaled_gains = self.apply_scaling_law(
                    source_gains, source_params, target_params
                )
                
                fit_scaled_sum = 0
                for esc in self.escenarios:
                    fit, _ = self.evaluar_pid(scaled_gains, esc, target_params)
                    fit_scaled_sum += fit
                fit_scaled = fit_scaled_sum / len(self.escenarios)
                
                # Calculate metrics
                degradation_unscaled = (fit_unscaled - fit_optimal) / fit_optimal * 100
                degradation_scaled = (fit_scaled - fit_optimal) / fit_optimal * 100
                
                # Retained benefit
                retained = (fit_zn - fit_scaled) / (fit_zn - fit_optimal) * 100
                
                transfer_results.append({
                    'source_factor': source_factor,
                    'target_factor': target_factor,
                    'source_name': self.variants[source_factor]['name'],
                    'target_name': self.variants[target_factor]['name'],
                    'fit_optimal': fit_optimal,
                    'fit_zn': fit_zn,
                    'fit_unscaled': fit_unscaled,
                    'fit_scaled': fit_scaled,
                    'degradation_unscaled': degradation_unscaled,
                    'degradation_scaled': degradation_scaled,
                    'retained_benefit': retained
                })
        
        return transfer_results
    
    def apply_scaling_law(self, gains, source_params, target_params):
        """Apply theoretical scaling law to gains"""
        # Extract parameters
        m_src = source_params['m']
        l_src = source_params['l']
        m_tgt = target_params['m']
        l_tgt = target_params['l']
        
        # Mass ratio
        m_ratio = m_tgt / m_src
        
        # Length ratio
        l_ratio = l_tgt / l_src
        
        # Apply scaling to each gain group
        scaled = gains.copy()
        
        # Altitude gains (indices 0-2)
        scaled[0] = gains[0] * m_ratio / l_ratio           # Kp
        scaled[1] = gains[1] * m_ratio / (l_ratio ** 1.5)  # Ki
        scaled[2] = gains[2] * m_ratio / np.sqrt(l_ratio)  # Kd
        
        # Roll gains (indices 3-5)
        scaled[3] = gains[3] * m_ratio / l_ratio
        scaled[4] = gains[4] * m_ratio / (l_ratio ** 1.5)
        scaled[5] = gains[5] * m_ratio / np.sqrt(l_ratio)
        
        # Pitch gains (indices 6-8)
        scaled[6] = gains[6] * m_ratio / l_ratio
        scaled[7] = gains[7] * m_ratio / (l_ratio ** 1.5)
        scaled[8] = gains[8] * m_ratio / np.sqrt(l_ratio)
        
        # Yaw gains (indices 9-11)
        scaled[9] = gains[9] * m_ratio / l_ratio
        scaled[10] = gains[10] * m_ratio / (l_ratio ** 1.5)
        scaled[11] = gains[11] * m_ratio / np.sqrt(l_ratio)
        
        return scaled
    
    def derive_scaling_exponents(self, optimal_gains):
        """Derive actual scaling exponents from optimal gains"""
        factors = [0.5, 1.0, 2.0, 5.0]
        masses = [self.variants[f]['params']['m'] for f in factors]
        lengths = [self.variants[f]['params']['l'] for f in factors]
        
        exponents = {}
        
        # Define gain groups
        groups = {
            'Kp_z': (0, 'altitude proportional'),
            'Ki_z': (1, 'altitude integral'),
            'Kd_z': (2, 'altitude derivative'),
            'Kp_phi': (3, 'roll proportional'),
            'Kp_theta': (6, 'pitch proportional'),
            'Kp_psi': (9, 'yaw proportional')
        }
        
        for name, (idx, desc) in groups.items():
            gains = [optimal_gains[f][idx] for f in factors]
            
            # Fit to power law: K = A * m^α * l^β
            # But m and l are correlated, so fit to m only and adjust
            
            # Log transform for linear fitting
            log_m = np.log(masses)
            log_g = np.log(gains)
            
            # Fit to mass only first
            coeffs_m = np.polyfit(log_m, log_g, 1)
            alpha = coeffs_m[0]
            
            # Then adjust for length
            # l ∝ m^(1/3) from geometric similarity
            # So observed exponent α_obs = α + β/3
            # We can estimate β = 3(α_obs - α_theory_with_length)
            
            exponents[name] = {
                'alpha_obs': alpha,
                'R2_mass': np.corrcoef(log_m, log_g)[0, 1] ** 2
            }
        
        return exponents
    
    # =========================================================================
    # FIGURE GENERATION
    # =========================================================================
    
    def generate_figures(self, results, transfer_results):
        """Generate all figures for the paper"""
        print("\n4. GENERATING FIGURES")
        
        self.fig1_kp_scaling(results)
        self.fig2_performance_comparison(transfer_results)
        self.fig3_scenario_breakdown(results)
        
        print("   ✓ All figures generated")
    
    def fig1_kp_scaling(self, results):
        """Figure 1: Kp scaling with mass"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        factors = [0.5, 1.0, 2.0, 5.0]
        masses = [self.variants[f]['params']['m'] for f in factors]
        
        # Altitude Kp
        kp_z = [results[f]['gains'][0] for f in factors]
        axes[0].scatter(masses, kp_z, s=100, color='#1f77b4', zorder=5)
        
        # Fit power law: Kp = a * m^b
        log_m = np.log(masses)
        log_kp = np.log(kp_z)
        coeffs = np.polyfit(log_m, log_kp, 1)
        b, log_a = coeffs[0], coeffs[1]
        a = np.exp(log_a)
        
        m_fit = np.linspace(0.5, 5, 100)
        kp_fit = a * m_fit ** b
        
        axes[0].plot(m_fit, kp_fit, 'r-', linewidth=2, 
                    label=f'$K_p \\propto m^{{{b:.2f}}}$')
        axes[0].set_xlabel('Mass (kg)', fontsize=12)
        axes[0].set_ylabel('$K_p$ (Altitude)', fontsize=12)
        axes[0].set_title('(a) Proportional Gain Scaling', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Attitude Kp (average of roll, pitch, yaw)
        kp_phi = [results[f]['gains'][3] for f in factors]
        kp_theta = [results[f]['gains'][6] for f in factors]
        kp_psi = [results[f]['gains'][9] for f in factors]
        kp_att = np.mean([kp_phi, kp_theta, kp_psi], axis=0)
        
        axes[1].scatter(masses, kp_att, s=100, color='#ff7f0e', zorder=5)
        
        log_kp_att = np.log(kp_att)
        coeffs = np.polyfit(log_m, log_kp_att, 1)
        b_att = coeffs[0]
        
        kp_att_fit = a_att * m_fit ** b_att if 'a_att' in locals() else a * m_fit ** b_att
        
        axes[1].plot(m_fit, kp_att_fit, 'r-', linewidth=2,
                    label=f'$K_p \\propto m^{{{b_att:.2f}}}$')
        axes[1].set_xlabel('Mass (kg)', fontsize=12)
        axes[1].set_ylabel('$K_p$ (Attitude avg)', fontsize=12)
        axes[1].set_title('(b) Attitude Gains Scaling', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Comparison of all gain types
        kd_z = [results[f]['gains'][2] for f in factors]
        log_kd = np.log(kd_z)
        coeffs_kd = np.polyfit(log_m, log_kd, 1)
        b_kd = coeffs_kd[0]
        
        axes[2].scatter(masses, kp_z, s=80, label='$K_p$ (altitude)', alpha=0.7)
        axes[2].scatter(masses, kd_z, s=80, label='$K_d$ (altitude)', alpha=0.7)
        axes[2].scatter(masses, kp_att, s=80, label='$K_p$ (attitude)', alpha=0.7)
        
        axes[2].set_xlabel('Mass (kg)', fontsize=12)
        axes[2].set_ylabel('Gain Value', fontsize=12)
        axes[2].set_title('(c) Comparison of Gain Types', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.suptitle('Figure 1: Scaling of PID Gains with Quadrotor Mass', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        path = os.path.join(self.fig_dir, 'fig1_kp_scaling.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ fig1_kp_scaling.png")
    
    def fig2_performance_comparison(self, transfer_results):
        """Figure 2: Performance comparison for gain transfer"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data for transfers FROM baseline (1.0) to others
        baseline_to_others = [r for r in transfer_results 
                             if r['source_factor'] == 1.0 and r['target_factor'] != 1.0]
        
        target_factors = [r['target_factor'] for r in baseline_to_others]
        target_names = [self.variants[f]['name'] for f in target_factors]
        
        unscaled = [r['fit_unscaled'] for r in baseline_to_others]
        scaled = [r['fit_scaled'] for r in baseline_to_others]
        optimal = [r['fit_optimal'] for r in baseline_to_others]
        
        x = np.arange(len(target_factors))
        width = 0.25
        
        ax1.bar(x - width, unscaled, width, label='Unscaled Transfer',
                color='#d62728', alpha=0.8, edgecolor='black')
        ax1.bar(x, scaled, width, label='Scaled Transfer (Eq. 5)',
                color='#2ca02c', alpha=0.8, edgecolor='black')
        ax1.bar(x + width, optimal, width, label='Re-optimized',
                color='#1f77b4', alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Target Quadrotor Size', fontsize=12)
        ax1.set_ylabel('Fitness (lower is better)', fontsize=12)
        ax1.set_title('(a) Performance Comparison', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(target_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bar chart of degradation
        degradation_unscaled = [r['degradation_unscaled'] for r in baseline_to_others]
        degradation_scaled = [r['degradation_scaled'] for r in baseline_to_others]
        
        ax2.bar(x - width/2, degradation_unscaled, width, label='Unscaled',
                color='#d62728', alpha=0.8, edgecolor='black')
        ax2.bar(x + width/2, degradation_scaled, width, label='Scaled',
                color='#2ca02c', alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Target Quadrotor Size', fontsize=12)
        ax2.set_ylabel('Degradation (%)', fontsize=12)
        ax2.set_title('(b) Performance Degradation', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(target_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (u, s) in enumerate(zip(degradation_unscaled, degradation_scaled)):
            ax2.text(i - width/2, u + 1, f'{u:.1f}%', ha='center', va='bottom', fontsize=9)
            ax2.text(i + width/2, s + 1, f'{s:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Figure 2: Performance of Gain Transfer from Baseline (1×) Quadrotor',
                    fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        path = os.path.join(self.fig_dir, 'fig2_performance.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ fig2_performance.png")
    
    def fig3_scenario_breakdown(self, results):
        """Figure 3: Performance breakdown by flight scenario for Large quadrotor"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        large_factor = 5.0
        large_params = self.variants[large_factor]['params']
        large_optimal = results[large_factor]['gains']
        baseline_gains = results[1.0]['gains']
        
        # Apply scaling to baseline gains
        scaled_gains = self.apply_scaling_law(
            baseline_gains, 
            self.variants[1.0]['params'],
            large_params
        )
        
        scenario_fits_optimal = []
        scenario_fits_scaled = []
        scenario_fits_unscaled = []
        scenario_names = []
        
        for idx, (esc, esc_name) in enumerate(zip(self.escenarios, self.nombres_escenarios)):
            fit_opt, _ = self.evaluar_pid(large_optimal, esc, large_params)
            fit_scaled, _ = self.evaluar_pid(scaled_gains, esc, large_params)
            fit_unscaled, _ = self.evaluar_pid(baseline_gains, esc, large_params)
            
            scenario_fits_optimal.append(fit_opt)
            scenario_fits_scaled.append(fit_scaled)
            scenario_fits_unscaled.append(fit_unscaled)
            scenario_names.append(f"S{idx+1}")
        
        x = np.arange(len(scenario_names))
        width = 0.25
        
        ax.bar(x - width, scenario_fits_unscaled, width, label='Unscaled Transfer',
               color='#d62728', alpha=0.7, edgecolor='black')
        ax.bar(x, scenario_fits_scaled, width, label='Scaled Transfer',
               color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.bar(x + width, scenario_fits_optimal, width, label='Re-optimized',
               color='#1f77b4', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Flight Scenario', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Figure 3: Performance by Scenario for Large (5×) Quadrotor',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add scenario descriptions as secondary x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels(['E1', 'E2', 'E3', 'E4', 'E5'], fontsize=9)
        ax2.set_xlabel('Scenario Code', fontsize=10)
        
        plt.tight_layout()
        
        path = os.path.join(self.fig_dir, 'fig3_scenarios.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ fig3_scenarios.png")
    
    # =========================================================================
    # TABLE GENERATION
    # =========================================================================
    
    def generate_tables(self, results, transfer_results):
        """Generate all tables for the paper"""
        print("\n5. GENERATING TABLES")
        
        self.table1_size_variants()
        self.table2_scaling_exponents(results)
        self.table3_degradation(transfer_results)
        self.table4_efficiency()
        self.table5_retained_benefit(transfer_results)
        
        print("   ✓ All tables generated")
    
    def table1_size_variants(self):
        """Table 1: Quadrotor size variants"""
        data = []
        for factor in self.size_factors:
            v = self.variants[factor]
            p = v['params']
            data.append({
                'Variant': v['name'],
                'Mass (kg)': f"{p['m']:.1f}",
                'Arm length (m)': f"{p['l']:.3f}",
                '$I_{xx}, I_{yy}$ (kg·m²)': f"{p['Ixx']:.3f}",
                '$I_{zz}$ (kg·m²)': f"{p['Izz']:.3f}"
            })
        
        df = pd.DataFrame(data)
        self.save_table(df, 'tab1_size_variants', 
                       'Quadrotor size variants used in scaling analysis')
    
    def table2_scaling_exponents(self, results):
        """Table 2: Scaling exponents"""
        exponents = self.derive_scaling_exponents(results)
        
        data = [
            {'Gain': 'Kp (altitude)', 'Theoretical': '$m^1 l^{-1}$', 
             'Observed': f"$m^{{{exponents['Kp_z']['alpha_obs']:.2f}}}$", 
             '$R^2$': f"{exponents['Kp_z']['R2_mass']:.3f}"},
            {'Gain': 'Ki (altitude)', 'Theoretical': '$m^1 l^{-3/2}$', 
             'Observed': f"$m^{{{exponents['Ki_z']['alpha_obs']:.2f}}}$", 
             '$R^2$': f"{exponents['Ki_z']['R2_mass']:.3f}"},
            {'Gain': 'Kd (altitude)', 'Theoretical': '$m^1 l^{-1/2}$', 
             'Observed': f"$m^{{{exponents['Kd_z']['alpha_obs']:.2f}}}$", 
             '$R^2$': f"{exponents['Kd_z']['R2_mass']:.3f}"},
            {'Gain': 'Kp (roll avg)', 'Theoretical': '$m^1 l^{-1}$', 
             'Observed': f"$m^{{{exponents['Kp_phi']['alpha_obs']:.2f}}}$", 
             '$R^2$': f"{exponents['Kp_phi']['R2_mass']:.3f}"}
        ]
        
        df = pd.DataFrame(data)
        self.save_table(df, 'tab2_scaling_exponents', 
                       'Scaling exponents for PID gains')
    
    def table3_degradation(self, transfer_results):
        """Table 3: Performance degradation"""
        baseline_transfers = [r for r in transfer_results 
                             if r['source_factor'] == 1.0 and r['target_factor'] != 1.0]
        
        data = []
        for r in baseline_transfers:
            data.append({
                'Target Size': r['target_name'],
                'Unscaled Transfer': f"{r['degradation_unscaled']:.1f}\\%",
                'Scaled Transfer': f"{r['degradation_scaled']:.1f}\\%",
                'Improvement': f"{(1 - r['degradation_scaled']/r['degradation_unscaled'])*100:.1f}\\%"
            })
        
        # Add averages
        avg_unscaled = np.mean([r['degradation_unscaled'] for r in baseline_transfers])
        avg_scaled = np.mean([r['degradation_scaled'] for r in baseline_transfers])
        avg_imp = (1 - avg_scaled/avg_unscaled) * 100
        
        data.append({
            'Target Size': '\\textbf{Average}',
            'Unscaled Transfer': f'\\textbf{{{avg_unscaled:.1f}\\%}}',
            'Scaled Transfer': f'\\textbf{{{avg_scaled:.1f}\\%}}',
            'Improvement': f'\\textbf{{{avg_imp:.1f}\\%}}'
        })
        
        df = pd.DataFrame(data)
        self.save_table(df, 'tab3_degradation', 
                       'Performance degradation when transferring gains')
    
    def table4_efficiency(self):
        """Table 4: Computational efficiency"""
        # Times from thesis
        time_per_opt = 3.1  # hours
        
        data = [
            {'Approach': 'Full re-optimization (all sizes)', 
             'Optimizations Required': '4', 
             'Total Time': f'{4*time_per_opt:.1f} hours',
             '\\% of Full': '100\\%'},
            {'Approach': 'Baseline + scaling to others', 
             'Optimizations Required': '1', 
             'Total Time': f'{time_per_opt:.1f} hours',
             '\\% of Full': '25\\%'},
            {'Approach': 'One-time scaling', 
             'Optimizations Required': '1', 
             'Total Time': f'{time_per_opt:.1f} hours',
             '\\% of Full': '25\\%'}
        ]
        
        df = pd.DataFrame(data)
        self.save_table(df, 'tab4_efficiency', 
                       'Computational efficiency analysis')
    
    def table5_retained_benefit(self, transfer_results):
        """Table 5: Retained benefit"""
        baseline_transfers = [r for r in transfer_results 
                             if r['source_factor'] == 1.0 and r['target_factor'] != 1.0]
        
        data = []
        for r in baseline_transfers:
            # ZN fitness from results
            zn_fit = r['fit_zn']
            opt_fit = r['fit_optimal']
            scaled_fit = r['fit_scaled']
            
            data.append({
                'Target Size': r['target_name'],
                '$J_{\\text{ZN}}$': f"{zn_fit:.4f}",
                '$J_{\\text{optimal}}$': f"{opt_fit:.4f}",
                '$J_{\\text{scaled}}$': f"{scaled_fit:.4f}",
                'Retained Benefit': f"{r['retained_benefit']:.1f}\\%"
            })
        
        # Average
        avg_retained = np.mean([r['retained_benefit'] for r in baseline_transfers])
        data.append({
            'Target Size': '\\textbf{Average}',
            '$J_{\\text{ZN}}$': '—',
            '$J_{\\text{optimal}}$': '—',
            '$J_{\\text{scaled}}$': '—',
            'Retained Benefit': f'\\textbf{{{avg_retained:.1f}\\%}}'
        })
        
        df = pd.DataFrame(data)
        self.save_table(df, 'tab5_retained_benefit', 
                       'Retained benefit of scaled gains')
    
    def save_table(self, df, filename, caption):
        """Save table as CSV and LaTeX"""
        # CSV
        csv_path = os.path.join(self.tab_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        # LaTeX
        latex_path = os.path.join(self.tab_dir, f"{filename}.tex")
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        print(f"   ✓ {filename}.tex")
    
    def save_results(self, results, transfer_results):
        """Save all results to JSON"""
        # Convert to serializable format
        results_serializable = {}
        for factor, data in results.items():
            if isinstance(factor, float):
                results_serializable[str(factor)] = {
                    'name': data['name'],
                    'gains': data['gains'].tolist()
                }
        
        transfer_serializable = []
        for r in transfer_results:
            r_copy = r.copy()
            for key in ['fit_optimal', 'fit_zn', 'fit_unscaled', 'fit_scaled',
                       'degradation_unscaled', 'degradation_scaled', 'retained_benefit']:
                r_copy[key] = float(r_copy[key])
            transfer_serializable.append(r_copy)
        
        all_results = {
            'results': results_serializable,
            'transfer': transfer_serializable,
            'timestamp': datetime.now().isoformat()
        }
        
        path = os.path.join(self.data_dir, 'scaling_analysis_results.json')
        with open(path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Results saved to {path}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quadrotor PID Scaling Analysis')
    parser.add_argument('--rapido', action='store_true', 
                       help='Modo rápido para pruebas')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("QUADROTOR PID SCALING ANALYSIS - TRAJECTORIES 2026")
    print("="*60)
    
    try:
        analyzer = QuadrotorScalingAnalysis(modo_rapido=args.rapido)
        results, transfer_results = analyzer.run_full_analysis()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print(f"📁 Output directory: {analyzer.out_dir}")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()