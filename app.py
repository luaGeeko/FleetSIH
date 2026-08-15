import streamlit as st
import pandas as pd
import time
import pydeck as pdk
from PIL import Image
from core.simulator import FleetSimulator
from optimization.ai_policy import AIPolicyOptimizer

# --- Configuration & State Initialization ---
st.set_page_config(page_title="Smart Fleet Platform", page_icon="🚚", layout="wide")

# Initialize Experiment Tracker in Session State
if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []

# Initialize the Backend Simulator and AI Optimizer in Session State
if "sim" not in st.session_state:
    st.session_state.sim = FleetSimulator()
    
    # Explicitly load the strictly required V1 model path and inject it
    try:
        ai_opt = AIPolicyOptimizer(model_path="models/fleet_ppo_v1_advanced_500k.zip")
        st.session_state.sim.set_ai_optimizer(ai_opt)
    except Exception as e:
        st.error(f"Failed to load AI Model: {e}")
        
    st.session_state.sim.generate_deterministic_scenario()

if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False

# City Bounding Box (Jaipur Urban Logistics Zone)
LAT_MIN, LAT_MAX = 26.8200, 26.9600
LON_MIN, LON_MAX = 75.7200, 75.8800

def grid_to_geo(x: float, y: float):
    """Maps 0-100 simulation grid to real-world city coordinates."""
    lat = LAT_MIN + (x / 100.0) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (y / 100.0) * (LON_MAX - LON_MIN)
    return lat, lon

sim = st.session_state.sim
metrics = sim.get_metrics()

# --- Header ---
st.title("🚚 Smart Fleet Operations Center")
st.markdown("Real-time fleet optimization and intelligent decentralized coordination.")
st.divider()

# --- UI Tabs ---
tab_live, tab_eval = st.tabs(["🔴 Live Simulation (Spatial Demo)", "📊 Evaluation & Benchmarks"])

# ==========================================
# TAB 1: LIVE SIMULATION
# ==========================================
with tab_live:
    st.markdown("""
    Watch the AI policy react dynamically. Because the agents utilize spatial observation sharing 
    (accessing positions and zone statuses), the system can instantly triage localized demand surges and breakdowns without recalculating a rigid global matrix.
    """)
    
    # --- Top Metrics Row ---
    st.subheader(f"📊 Operations Overview (Tick: {metrics['time_step']})")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Available / Total", f"{metrics['available_vehicles']} / {metrics['total_vehicles']}", 
                  delta=f"-{metrics['broken_vehicles']} broken" if metrics['broken_vehicles'] > 0 else "0")
    with col2:
        st.metric("Moving Vehicles", metrics['moving_vehicles'])
    with col3:
        st.metric("Pending Shipments", metrics['pending_shipments'], 
                  delta=f"{metrics['delivered_shipments']} delivered" if metrics['delivered_shipments'] > 0 else "")
    with col4:
        st.metric("Fleet Utilization", f"{metrics['utilization']:.1f}%")

    st.divider()

    # --- Main Dashboard Area ---
    map_col, info_col = st.columns([2, 1])

    with map_col:
        st.subheader("📍 Live Fleet Map (Metro Zone)")
        
        # 1. Vehicles Data
        vehicle_data = []
        for v in sim.vehicles.values():
            lat, lon = grid_to_geo(v.position[0], v.position[1])
            vehicle_data.append({
                "lat": lat,
                "lon": lon,
                "vehicle_id": v.id,
                "status": v.status,
                "shipment_id": ", ".join(v.assigned_shipments) if v.assigned_shipments else "None"
            })
        vehicle_df = pd.DataFrame(vehicle_data)

        # 2. Shipments: Pickup + Destination Data
        pickup_data = []
        destination_data = []
        
        for s in sim.shipments.values():
            if s.status == "delivered":
                continue
                
            pickup_lat, pickup_lon = grid_to_geo(s.pickup[0], s.pickup[1])
            dest_lat, dest_lon = grid_to_geo(s.destination[0], s.destination[1])

            if s.status in ["pending", "assigned"]:
                pickup_data.append({
                    "lat": pickup_lat,
                    "lon": pickup_lon,
                    "shipment_id": s.id,
                    "status": s.status,
                    "vehicle_id": s.assigned_vehicle_id or "Unassigned"
                })

            destination_data.append({
                "lat": dest_lat,
                "lon": dest_lon,
                "shipment_id": s.id,
                "status": s.status,
                "vehicle_id": s.assigned_vehicle_id or "Unassigned"
            })
            
        pickup_df = pd.DataFrame(pickup_data)
        destination_df = pd.DataFrame(destination_data)

        # 3. Current Vehicle Routes Data
        route_data = []
        for v in sim.vehicles.values():
            if not v.route:
                continue
                
            path = []
            start_lat, start_lon = grid_to_geo(v.position[0], v.position[1])
            path.append([start_lon, start_lat])

            for stop in v.route:
                lat, lon = grid_to_geo(stop.location[0], stop.location[1])
                path.append([lon, lat])

            route_data.append({
                "vehicle_id": v.id,
                "path": path
            })
        route_df = pd.DataFrame(route_data)

        # 4. Map Layers
        layers = []

        if not vehicle_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=vehicle_df,
                get_position="[lon, lat]",
                get_radius=350,
                get_fill_color="[50, 150, 255, 220]", # Blue
                pickable=True,
            ))

        if not pickup_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=pickup_df,
                get_position="[lon, lat]",
                get_radius=220,
                get_fill_color="[255, 180, 0, 220]", # Orange
                pickable=True,
            ))
            
            layers.append(pdk.Layer(
                "TextLayer",
                data=pickup_df,
                get_position="[lon, lat]",
                get_text="'📦 ' + shipment_id",
                get_size=14,
                get_color="[255, 255, 255, 255]",
                get_text_anchor="'middle'",
                get_alignment_baseline="'bottom'",
            ))

        if not destination_df.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=destination_df,
                get_position="[lon, lat]",
                get_radius=180,
                get_fill_color="[80, 220, 120, 220]", # Green
                pickable=True,
            ))

        if not route_df.empty:
            layers.append(pdk.Layer(
                "PathLayer",
                data=route_df,
                get_path="path",
                get_width=5,
                get_color="[100, 180, 255, 180]", # Light Blue
                pickable=True,
            ))

        # 5. Fixed View & Render
        center_lat = (LAT_MIN + LAT_MAX) / 2
        center_lon = (LON_MIN + LON_MAX) / 2

        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11.5,
            pitch=0,
            bearing=0,
        )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": """
                <b>Vehicle:</b> {vehicle_id}
                <br/>
                <b>Shipment:</b> {shipment_id}
                <br/>
                <b>Status:</b> {status}
                """,
                "style": {
                    "backgroundColor": "#1f2937",
                    "color": "white",
                },
            },
        )

        st.pydeck_chart(deck, width="stretch")

        st.subheader("📈 Performance Impact (Shift-to-Date)")
        
        current_strategy = getattr(sim, "current_strategy", "Greedy Dispatch")
        
        comp_df = pd.DataFrame({
            "Metric": ["Completion Rate", "On-time Delivery", "Fleet Utilization", "Total Distance"],
            "Greedy Baseline": [
                f"{metrics['completion_rate']:.1f}%" if current_strategy == "Greedy Dispatch" else "--", 
                f"{metrics['on_time_rate']:.1f}%" if current_strategy == "Greedy Dispatch" else "--", 
                f"{metrics['utilization']:.1f}%" if current_strategy == "Greedy Dispatch" else "--", 
                f"{metrics['total_distance']:.1f} km" if current_strategy == "Greedy Dispatch" else "--"
            ],
            "OR-Tools": [
                f"{metrics['completion_rate']:.1f}%" if current_strategy == "OR-Tools CVRP" else "--", 
                f"{metrics['on_time_rate']:.1f}%" if current_strategy == "OR-Tools CVRP" else "--", 
                f"{metrics['utilization']:.1f}%" if current_strategy == "OR-Tools CVRP" else "--", 
                f"{metrics['total_distance']:.1f} km" if current_strategy == "OR-Tools CVRP" else "--"
            ],
            "AI Coordinator": [
                f"{metrics['completion_rate']:.1f}%" if current_strategy == "AI Coordinator" else "--", 
                f"{metrics['on_time_rate']:.1f}%" if current_strategy == "AI Coordinator" else "--", 
                f"{metrics['utilization']:.1f}%" if current_strategy == "AI Coordinator" else "--", 
                f"{metrics['total_distance']:.1f} km" if current_strategy == "AI Coordinator" else "--"
            ]
        })
        st.dataframe(comp_df, hide_index=True, width="stretch")

    with info_col:
        st.subheader("⚙️ Control Panel")

        strategy = st.radio("Routing Strategy", ["Greedy Dispatch", "OR-Tools CVRP", "AI Coordinator"], horizontal=True)
        sim.set_strategy(strategy)
        
        if st.button("▶️ Run Simulation" if not st.session_state.simulation_running else "⏸️ Pause Simulation", width="stretch"):
            st.session_state.simulation_running = not st.session_state.simulation_running
            st.rerun()
            
        st.markdown("### Inject Disruptions")
        dist1, dist2, dist3 = st.columns(3)
        with dist1:
            if st.button("🚨 Traffic", width="stretch"):
                sim.inject_traffic("Zone_A", severity=0.5)
                st.rerun()
        with dist2:
            if st.button("🔧 Breakdown", width="stretch"):
                sim.inject_breakdown("V01")
                st.rerun()
        with dist3:
            if st.button("📦 Surge", width="stretch"):
                sim.inject_demand_surge(num_new=5)
                st.rerun()
                
        if st.button("🔄 Reset Environment", width="stretch"):
            sim.generate_deterministic_scenario()
            st.session_state.simulation_running = False
            st.rerun()

        st.divider()

        st.subheader("📡 System Alerts & AI Action")
        
        if not metrics['alerts']:
            st.success("✅ Fleet operating normally.")
            st.info("💡 AI: No active rerouting recommendations.")
        else:
            for alert in reversed(metrics['alerts'][-3:]):
                if "CRITICAL" in alert:
                    st.error(alert)
                    st.info("💡 **Decision Engine:** Reassigning pending shipments from broken vehicle.")
                elif "TRAFFIC" in alert:
                    st.warning(alert)
                    st.info("💡 **Decision Engine:** Rerouting active fleet to avoid traffic zone.")
                elif "SURGE" in alert:
                    st.warning(alert)
                    st.info("💡 **Decision Engine:** Dispatching idle fleet to cover demand surge.")

    # --- Logs ---
    st.divider()
    with st.expander("Terminal & Optimizer Logs", expanded=True):
        if st.session_state.simulation_running:
            log_text = f"""[{time.strftime('%H:%M:%S')}] Tick {metrics['time_step']}: System actively monitoring fleet.
[{time.strftime('%H:%M:%S')}] Evaluated {metrics['pending_shipments']} pending shipments against {metrics['available_vehicles']} available vehicles."""
            st.code(log_text, language="text")
        else:
            st.code("System Idle. Waiting for simulation start.", language="text")

    # --- EXPERIMENT TRACKER & EXPORT ---
    st.divider()
    st.subheader("🧪 Live Session Tracker")
    st.write("Save your current simulation results to compare strategies visually.")

    # Input for saving the run
    col_name, col_btn = st.columns([3, 1])
    with col_name:
        scenario_name = st.text_input("Scenario Name (e.g., 'Heavy Traffic + Surge')", value="Baseline")
    with col_btn:
        st.write("") 
        st.write("") 
        if st.button("💾 Save Current Run", width="stretch"):
            st.session_state.experiment_history.append({
                "Scenario": scenario_name,
                "Strategy": current_strategy,
                "Completion Rate (%)": round(metrics['completion_rate'], 1),
                "On-Time Rate (%)": round(metrics['on_time_rate'], 1),
                "Utilization (%)": round(metrics['utilization'], 1),
                "Total Distance (km)": round(metrics['total_distance'], 1)
            })
            st.success(f"Saved {current_strategy} results!")

    # Display the Leaderboard if it has data
    if st.session_state.experiment_history:
        history_df = pd.DataFrame(st.session_state.experiment_history)
        
        st.dataframe(history_df, width="stretch", hide_index=True)

        # Download / Clear Buttons
        dl_col, clear_col = st.columns([2, 2])
        with dl_col:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Session CSV",
                data=csv_data,
                file_name='smart_fleet_live_session.csv',
                mime='text/csv',
                width="stretch"
            )
        with clear_col:
            if st.button("🗑️ Clear History", width="stretch"):
                st.session_state.experiment_history = []
                st.rerun()


# ==========================================
# TAB 2: SCIENTIFIC BENCHMARK REPORT
# ==========================================
with tab_eval:
    st.header("Offline Resilience & Scalability Benchmarks")
    
    st.markdown("""
    ### 🔬 Scientific Validation & Results
    This dashboard presents the results of our multi-seed controlled offline experiments. We compare the AI Coordinator against traditional Greedy Dispatch and global OR-Tools optimization to evaluate performance across varied operational conditions.
    
    *   **The Adaptation Cost:** During demand surges and chaotic disruptions, we measure the extra distance incurred to handle the dynamic constraints.
    *   **Reliability Under Stress:** We evaluate how well each strategy balances static routing efficiency against strict service-level constraints (like on-time delivery rates) during concurrent systemic disruptions.
    *   **Agility & Recovery:** By tracking the average ticks to recover, we benchmark how quickly the system reassigns resources following a vehicle breakdown or sudden demand surge.
    """)
    
    st.divider()
    
    # Render the statically generated dashboard
    try:
        dash_image = Image.open("evaluation/results/performance_dashboard_v1.png") 
        st.image(dash_image, caption="Multi-Seed Controlled Evaluation (10-Vehicle Fleet, 500k Timesteps)", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Could not find the V1 performance dashboard image. Please run the offline benchmark plotting script (`python -m evaluation.plot_results --version v1`) first.")

# --- Auto-Advance Simulation Loop (Outside tabs so it runs globally) ---
if st.session_state.simulation_running:
    sim.step()
    time.sleep(0.5) 
    st.rerun()