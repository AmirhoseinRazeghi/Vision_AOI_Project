import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime

# ---------------------------------------------------------
# PAGE CONFIGURATION & CYBER-INDUSTRIAL CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="CyberVision AOI | Industry 4.0 Machine Vision & PLC Twin",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #080b11; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 800; color: #00f2fe; font-family: monospace; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #90a4ae; text-transform: uppercase; letter-spacing: 0.5px; }
    .hud-banner-pass {
        background: linear-gradient(90deg, rgba(0,230,118,0.15) 0%, rgba(14,17,23,0.9) 100%);
        border-left: 5px solid #00e676; border-right: 1px solid #1e3a2f;
        padding: 12px 18px; border-radius: 6px; color: #00e676; font-family: monospace; font-weight: bold; font-size: 1.05rem;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.15); margin-bottom: 12px;
    }
    .hud-banner-fail {
        background: linear-gradient(90deg, rgba(255,23,68,0.22) 0%, rgba(14,17,23,0.9) 100%);
        border-left: 5px solid #ff1744; border-right: 1px solid #4a151b;
        padding: 12px 18px; border-radius: 6px; color: #ff1744; font-family: monospace; font-weight: bold; font-size: 1.05rem;
        box-shadow: 0 0 18px rgba(255, 23, 68, 0.25); margin-bottom: 12px;
    }
    .plc-rack {
        background-color: #101520; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;
        font-family: monospace; font-size: 0.88rem; color: #e2e8f0;
    }
    .led-on { color: #00e676; font-weight: bold; text-shadow: 0 0 8px #00e676; }
    .led-fault { color: #ff1744; font-weight: bold; text-shadow: 0 0 8px #ff1744; }
    .led-off { color: #475569; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE FOR PRODUCTION TELEMETRY & LOGS
# ---------------------------------------------------------
if "total_inspected" not in st.session_state:
    st.session_state.total_inspected = 124
    st.session_state.pass_count = 116
    st.session_state.reject_count = 8
    st.session_state.defect_tally = {
        "Solder Bridge Short": 3,
        "Missing SMD Capacitor": 2,
        "Thermal Hotspot (>85°C)": 2,
        "IC Lead Misalignment": 1
    }
    st.session_state.plc_logs = [
        {"Timestamp": "21:40:12.104", "Unit_ID": "PCB-0121", "Result": "PASS", "Defect_Type": "None", "PLC_Action": "Q0.0 Conveyor Continue", "Confidence": "99.4%"},
        {"Timestamp": "21:40:15.480", "Unit_ID": "PCB-0122", "Result": "REJECT", "Defect_Type": "Solder Bridge Short", "PLC_Action": "Q0.1 Pneumatic Ejector ON", "Confidence": "98.7%"},
        {"Timestamp": "21:40:18.912", "Unit_ID": "PCB-0123", "Result": "PASS", "Defect_Type": "None", "PLC_Action": "Q0.0 Conveyor Continue", "Confidence": "99.1%"},
        {"Timestamp": "21:40:22.305", "Unit_ID": "PCB-0124", "Result": "PASS", "Defect_Type": "None", "PLC_Action": "Q0.0 Conveyor Continue", "Confidence": "99.6%"},
    ]

# ---------------------------------------------------------
# SIDEBAR: VISION ENGINE & PLC PARAMETERS
# ---------------------------------------------------------
st.sidebar.title("👁️ CyberVision AOI Panel")
st.sidebar.caption("PLC Interface: Siemens S7-1500 PROFINET | Eng. Amirhosein Razeghi")

camera_source = st.sidebar.radio(
    "📷 Image Acquisition Source",
    ["Conveyor Digital Twin (Synthetic PCB)", "Live Hardware Webcam (OpenCV)"]
)

vision_mode = st.sidebar.selectbox(
    "🔍 Optical Processing Filter",
    [
        "Dual-Camera Split (Optical HUD + Thermal IR)",
        "RGB Optical + AI Bounding HUD",
        "Canny Structural Edge & Contours",
        "Thermal Infrared (IR) Hotspot Map"
    ]
)

st.sidebar.subheader("1. Defect Injection & Fault Test")
defect_scenario = st.sidebar.selectbox(
    "Simulated PCB Condition",
    [
        "Golden Sample (100% Healthy PCB)",
        "Inject: Solder Bridge Short (IC Pins)",
        "Inject: Missing SMD Capacitor (C14)",
        "Inject: MOSFET Thermal Overheat (>95°C)",
        "Inject: Multi-Fault Critical Reject"
    ]
)

st.sidebar.subheader("2. OpenCV Vision Hyperparameters")
canny_low = st.sidebar.slider("Canny Edge Threshold Low", 20, 150, 60, 5)
canny_high = st.sidebar.slider("Canny Edge Threshold High", 100, 300, 160, 10)
conf_thresh = st.sidebar.slider("AI Defect Confidence Threshold (%)", 70.0, 99.0, 88.0, 1.0)
thermal_limit = st.sidebar.slider("IR Thermal Alarm Limit (°C)", 60.0, 110.0, 85.0, 2.5)

# ---------------------------------------------------------
# SYNTHETIC HIGH-RES INDUSTRIAL PCB GENERATOR & OPENCV ENGINE
# ---------------------------------------------------------
def generate_and_inspect_pcb(scenario, c_low, c_high, t_limit):
    h, w = 540, 900
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Dark Conveyor Belt Background
    img[:] = (22, 26, 32)
    for y in range(0, h, 20):
        cv2.line(img, (0, y), (w, y), (28, 33, 40), 1)

    # PCB Substrate (Deep Industrial Emerald Green)
    px1, py1, px2, py2 = 90, 50, 810, 490
    cv2.rectangle(img, (px1, py1), (px2, py2), (38, 82, 45), -1)
    cv2.rectangle(img, (px1, py1), (px2, py2), (80, 190, 100), 3)

    # Mounting Holes & Fiducial Alignment Marks
    for cx, cy in [(125, 85), (775, 85), (125, 455), (775, 455)]:
        cv2.circle(img, (cx, cy), 14, (160, 200, 220), 3)
        cv2.circle(img, (cx, cy), 8, (20, 24, 30), -1)
        cv2.line(img, (cx - 22, cy), (cx + 22, cy), (0, 242, 254), 1)
        cv2.line(img, (cx, cy - 22), (cx, cy + 22), (0, 242, 254), 1)

    # Gold Copper Traces
    traces = [
        [(200, 150), (370, 150), (410, 210)],
        [(200, 200), (350, 200), (410, 240)],
        [(530, 230), (650, 230), (680, 160)],
        [(530, 280), (660, 280), (660, 390)],
        [(450, 340), (450, 430), (280, 430)]
    ]
    for tr in traces:
        pts = np.array(tr, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (85, 175, 215), 3, cv2.LINE_AA)

    # Central Microcontroller QFP Package (STM32F4)
    ic_x1, ic_y1, ic_x2, ic_y2 = 390, 190, 530, 330
    cv2.rectangle(img, (ic_x1, ic_y1), (ic_x2, ic_y2), (35, 38, 42), -1)
    cv2.rectangle(img, (ic_x1, ic_y1), (ic_x2, ic_y2), (90, 95, 105), 2)
    cv2.circle(img, (ic_x1 + 18, ic_y1 + 18), 5, (180, 180, 180), -1)
    cv2.putText(img, "STM32F407VG", (ic_x1 + 16, ic_y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(img, "ARM CORTEX-M4", (ic_x1 + 18, ic_y1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 170), 1, cv2.LINE_AA)

    # IC Pins
    for offset in range(15, 130, 14):
        cv2.rectangle(img, (ic_x1 - 14, ic_y1 + offset), (ic_x1, ic_y1 + offset + 6), (200, 210, 215), -1)
        cv2.rectangle(img, (ic_x2, ic_y1 + offset), (ic_x2 + 14, ic_y1 + offset + 6), (200, 210, 215), -1)
        cv2.rectangle(img, (ic_x1 + offset, ic_y1 - 14), (ic_x1 + offset + 6, ic_y1), (200, 210, 215), -1)
        cv2.rectangle(img, (ic_x1 + offset, ic_y2), (ic_x1 + offset + 6, ic_y2 + 14), (200, 210, 215), -1)

    # Power MOSFET Module
    cv2.rectangle(img, (630, 120), (735, 215), (30, 32, 36), -1)
    cv2.rectangle(img, (645, 105), (720, 120), (180, 190, 200), -1)
    cv2.putText(img, "IRFZ44N", (648, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # SMD Capacitors Bank
    smd_positions = [("C12", 210, 135), ("C13", 210, 195), ("C14", 210, 255)]
    for label, sx, sy in smd_positions:
        if label == "C14" and ("Missing SMD" in scenario or "Multi-Fault" in scenario):
            cv2.rectangle(img, (sx, sy), (sx + 14, sy + 28), (170, 180, 185), -1)
            cv2.rectangle(img, (sx + 46, sy), (sx + 60, sy + 28), (170, 180, 185), -1)
        else:
            cv2.rectangle(img, (sx, sy), (sx + 14, sy + 28), (200, 210, 215), -1)
            cv2.rectangle(img, (sx + 46, sy), (sx + 60, sy + 28), (200, 210, 215), -1)
            cv2.rectangle(img, (sx + 12, sy + 2), (sx + 48, sy + 26), (70, 115, 165), -1)
        cv2.putText(img, label, (sx - 35, sy + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 230, 190), 1, cv2.LINE_AA)

    # Inject Solder Bridge Defect
    if "Solder Bridge" in scenario or "Multi-Fault" in scenario:
        cv2.circle(img, (ic_x1 - 7, ic_y1 + 50), 10, (220, 230, 240), -1)
        cv2.rectangle(img, (ic_x1 - 15, ic_y1 + 41), (ic_x1 + 2, ic_y1 + 63), (235, 240, 245), -1)

    # 1. Canny Structural Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, c_low, c_high)
    edge_colored = np.zeros_like(img)
    edge_colored[edges > 0] = (254, 242, 0)

    # 2. Synthetic Thermal Infrared Map
    thermal_gray = np.full((h, w), 45, dtype=np.uint8)
    cv2.circle(thermal_gray, (460, 260), 95, 135, -1)
    mosfet_intensity = 245 if ("Thermal" in scenario or "Multi-Fault" in scenario) else 155
    cv2.circle(thermal_gray, (682, 168), 85, mosfet_intensity, -1)
    if "Solder Bridge" in scenario or "Multi-Fault" in scenario:
        cv2.circle(thermal_gray, (ic_x1 - 7, ic_y1 + 50), 55, 230, -1)
    thermal_blur = cv2.GaussianBlur(thermal_gray, (65, 65), 0)
    thermal_map = cv2.applyColorMap(thermal_blur, cv2.COLORMAP_JET)
    thermal_hud = cv2.addWeighted(thermal_map, 0.78, img, 0.22, 0)

    # 3. AI HUD & Defect Annotation Overlay
    hud_img = img.copy()
    defects_found = []
    peak_temp = 64.2

    cv2.rectangle(hud_img, (ic_x1 - 20, ic_y1 - 20), (ic_x2 + 20, ic_y2 + 20), (118, 230, 0), 1)
    cv2.putText(hud_img, "MCU_U1: OK (99.6%)", (ic_x1 - 18, ic_y1 - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (118, 230, 0), 1, cv2.LINE_AA)

    if "Solder Bridge" in scenario or "Multi-Fault" in scenario:
        bx1, by1, bx2, by2 = ic_x1 - 30, ic_y1 + 28, ic_x1 + 15, ic_y1 + 75
        cv2.rectangle(hud_img, (bx1, by1), (bx2, by2), (68, 23, 255), 2)
        cv2.circle(hud_img, (ic_x1 - 7, ic_y1 + 50), 24, (68, 23, 255), 1)
        cv2.putText(hud_img, "NG: SOLDER BRIDGE [98.9%]", (bx1 - 140, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (68, 23, 255), 2, cv2.LINE_AA)
        defects_found.append(("Solder Bridge Short", 98.9, "X:37.6mm Y:24.1mm"))
        peak_temp = max(peak_temp, 88.4)

    if "Missing SMD" in scenario or "Multi-Fault" in scenario:
        bx1, by1, bx2, by2 = 198, 245, 282, 292
        cv2.rectangle(hud_img, (bx1, by1), (bx2, by2), (68, 23, 255), 2)
        cv2.putText(hud_img, "NG: MISSING C14 [99.2%]", (bx1 - 20, by2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (68, 23, 255), 2, cv2.LINE_AA)
        defects_found.append(("Missing SMD Capacitor", 99.2, "X:21.4mm Y:26.8mm"))

    if "Thermal" in scenario or "Multi-Fault" in scenario:
        peak_temp = 98.7
    
    if peak_temp >= t_limit:
        bx1, by1, bx2, by2 = 615, 95, 750, 230
        cv2.rectangle(hud_img, (bx1, by1), (bx2, by2), (0, 145, 255), 2)
        cv2.putText(hud_img, f"IR HOTSPOT: {peak_temp:.1f}C", (bx1 - 10, by2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 145, 255), 2, cv2.LINE_AA)
        cv2.circle(thermal_hud, (682, 168), 42, (255, 255, 255), 2)
        cv2.putText(thermal_hud, f"MAX: {peak_temp:.1f} C [ALARM]", (595, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        defects_found.append(("Thermal Hotspot (>85°C)", 97.5, "X:68.2mm Y:16.8mm"))
    else:
        cv2.putText(thermal_hud, f"MAX TEMP: {peak_temp:.1f} C [NORMAL]", (595, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 255, 220), 1, cv2.LINE_AA)

    for target in (hud_img, edge_colored, thermal_hud):
        cv2.line(target, (20, 20), (65, 20), (254, 242, 0), 2)
        cv2.line(target, (20, 20), (20, 65), (254, 242, 0), 2)
        cv2.line(target, (w - 65, 20), (w - 20, 20), (254, 242, 0), 2)
        cv2.line(target, (w - 20, 20), (w - 20, 65), (254, 242, 0), 2)
        cv2.line(target, (20, h - 20), (65, h - 20), (254, 242, 0), 2)
        cv2.line(target, (20, h - 65), (20, h - 20), (254, 242, 0), 2)
        cv2.line(target, (w - 65, h - 20), (w - 20, h - 20), (254, 242, 0), 2)
        cv2.line(target, (w - 20, h - 65), (w - 20, h - 20), (254, 242, 0), 2)
        
        cv2.rectangle(target, (0, 0), (w, 32), (12, 16, 23), -1)
        cv2.putText(target, "CAM-01: TELECENTRIC 5MP | FPS: 60.0 | EXP: 1.2ms | CAL: 12.5 um/px", (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (254, 242, 0), 1, cv2.LINE_AA)
        status_txt = "STATUS: PASS [OK]" if len(defects_found) == 0 else f"STATUS: REJECT [{len(defects_found)} FAULTS]"
        status_col = (118, 230, 0) if len(defects_found) == 0 else (68, 23, 255)
        cv2.putText(target, status_txt, (w - 255, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 2, cv2.LINE_AA)

    return (
        cv2.cvtColor(hud_img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(edge_colored, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(thermal_hud, cv2.COLOR_BGR2RGB),
        defects_found,
        peak_temp
    )

# ---------------------------------------------------------
# HEADER & AUTOMATED BATCH TRIGGER
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([3.3, 1.2])
with col_h1:
    st.title("👁️ CyberVision AOI — Industry 4.0 Optical Inspection & PLC Twin")
    st.caption("Architecture: OpenCV Machine Vision | IR Thermography | Siemens S7 PROFINET I/O | Real-Time OEE Analytics")

with col_h2:
    st.write("")
    run_batch = st.button("▶️ Start Automated Conveyor Batch", use_container_width=True)

# ---------------------------------------------------------
# RUN INSPECTION ENGINE
# ---------------------------------------------------------
rgb_hud, edge_hud, ir_hud, active_defects, max_ir_temp = generate_and_inspect_pcb(
    defect_scenario, canny_low, canny_high, thermal_limit
)
is_pass = len(active_defects) == 0

if is_pass:
    st.markdown(
        '<div class="hud-banner-pass">🟢 PLC PROFINET STATUS: UNIT PASSED QUALITY CHECK | Q0.0 CONVEYOR RUN | Q0.1 PNEUMATIC EJECTOR: IDLE (OFF)</div>',
        unsafe_allow_html=True
    )
else:
    fault_names = ", ".join([d[0] for d in active_defects])
    st.markdown(
        f'<div class="hud-banner-fail">🚨 PLC REJECT INTERLOCK TRIGGERED [Q0.1 HIGH-SPEED SOLENOID ACTIVE] — DETECTED: {fault_names.upper()}</div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# OEE & PRODUCTION TELEMETRY METRICS ROW
# ---------------------------------------------------------
total_u = st.session_state.total_inspected
pass_u = st.session_state.pass_count
rej_u = st.session_state.reject_count
fpy = (pass_u / total_u) * 100.0 if total_u > 0 else 100.0
oee_score = fpy * 0.965

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Inspected", f"{total_u} Units", "+1 Cycle")
m2.metric("First Pass Yield (FPY)", f"{fpy:.1f} %", f"{pass_u} OK / {rej_u} NG")
m3.metric("OEE Index", f"{oee_score:.1f} %", "World-Class > 85%")
m4.metric("Vision Cycle Time", "14.2 ms", "60 FPS Lock")
m5.metric("Peak Component Temp", f"{max_ir_temp:.1f} °C", f"Limit: {thermal_limit:.0f} °C")
m6.metric("PLC Reject Valve (Q0.1)", "ACTIVE (ON)" if not is_pass else "STANDBY (OFF)", "24V DC Solenoid")

st.divider()

# ---------------------------------------------------------
# MAIN VIEWPORT & SIEMENS S7-1500 PLC I/O RACK
# ---------------------------------------------------------
viewport_placeholder = st.empty()

def render_main_station(rgb_v, edge_v, ir_v, passed_flag, defects_list, temp_val, chart_key="oee_static"):
    col_cam, col_plc = st.columns([2.65, 1.35])
    
    with col_cam:
        if camera_source == "Live Hardware Webcam (OpenCV)":
            st.info("📷 Hardware Webcam Mode Active: Capture a frame below for live OpenCV Canny Edge & Contour inspection.")
            cam_photo = st.camera_input("Take Snapshot from Assembly Line Camera", key=f"cam_{chart_key}")
            if cam_photo is not None:
                file_bytes = np.asarray(bytearray(cam_photo.read()), dtype=np.uint8)
                webcam_img = cv2.imdecode(file_bytes, 1)
                gray_w = cv2.cvtColor(webcam_img, cv2.COLOR_BGR2GRAY)
                edges_w = cv2.Canny(gray_w, canny_low, canny_high)
                contours, _ = cv2.findContours(edges_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(webcam_img, contours, -1, (0, 242, 254), 2)
                st.image(cv2.cvtColor(webcam_img, cv2.COLOR_BGR2RGB), caption="Live Webcam OpenCV Contour Inspection", use_container_width=True)
        else:
            if vision_mode == "Dual-Camera Split (Optical HUD + Thermal IR)":
                c_left, c_right = st.columns(2)
                with c_left:
                    st.image(rgb_v, caption="CAM-01: RGB Optical Inspection & AI Bounding HUD", use_container_width=True)
                with c_right:
                    st.image(ir_v, caption="CAM-02: LWIR Thermal Hotspot Map (COLORMAP_JET)", use_container_width=True)
            elif vision_mode == "RGB Optical + AI Bounding HUD":
                st.image(rgb_v, caption="CAM-01: High-Resolution RGB Optical HUD", use_container_width=True)
            elif vision_mode == "Canny Structural Edge & Contours":
                st.image(edge_v, caption="OpenCV Canny Structural Gradient & Edge Map", use_container_width=True)
            else:
                st.image(ir_v, caption="CAM-02: Infrared Thermography Inspection", use_container_width=True)

    with col_plc:
        st.markdown("##### 🎛️ Siemens S7-1500 PROFINET I/O Rack")
        q00_cls = "led-on" if passed_flag else "led-off"
        q00_txt = "ON (24V)" if passed_flag else "OFF (0V)"
        q01_cls = "led-fault" if not passed_flag else "led-off"
        q01_txt = "TRIPPED (24V)" if not passed_flag else "OFF (0V)"
        reg_code = "0x0000 (OK)" if passed_flag else f"0x0{len(defects_list)}0F (FAULT)"

        st.markdown(f"""
        <div class="plc-rack">
            <div style="border-bottom: 1px solid #334155; padding-bottom: 6px; margin-bottom: 8px; color: #38bdf8; font-weight: bold;">
                CPU 1516-3 PN/DP | IP: 192.168.0.10
            </div>
            <div>• <b>I0.0</b> [Optical Part Sensor]: <span class="led-on">● HIGH (1)</span></div>
            <div>• <b>I0.1</b> [Ring Strobe Ready]: <span class="led-on">● HIGH (1)</span></div>
            <div>• <b>Q0.0</b> [Conveyor VFD Run]: <span class="{q00_cls}">● {q00_txt}</span></div>
            <div>• <b>Q0.1</b> [Pneumatic Ejector]: <span class="{q01_cls}">● {q01_txt}</span></div>
            <div>• <b>Q0.2</b> [Andon Red Tower]: <span class="{q01_cls}">● {q01_txt}</span></div>
            <div style="border-top: 1px solid #334155; margin-top: 10px; padding-top: 8px; color: #94a3b8;">
                <b>MODBUS TCP HOLDING REGISTERS:</b><br>
                • <b>40001</b> [Status Word]: <code>{reg_code}</code><br>
                • <b>40002</b> [Peak Temp]: <code>{int(temp_val * 10)} (x0.1 °C)</code><br>
                • <b>40003</b> [FPY Yield]: <code>{int(fpy * 10)} (x0.1 %)</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini OEE Gauge with unique key
        fig_oee = go.Figure(go.Indicator(
            mode="gauge+number",
            value=oee_score,
            number={'suffix': "%", 'font': {'size': 26, 'color': '#00f2fe'}},
            title={'text': "Real-Time Line OEE", 'font': {'size': 14, 'color': '#cbd5e1'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': "#00e676" if oee_score >= 85 else "#ffab00"},
                'bgcolor': "#141a26",
                'steps': [
                    {'range': [0, 70], 'color': "#3b1219"},
                    {'range': [70, 85], 'color': "#3b2f12"},
                    {'range': [85, 100], 'color': "#0f3828"}
                ]
            }
        ))
        fig_oee.update_layout(height=185, margin=dict(l=18, r=18, t=35, b=10), paper_bgcolor="#080b11", font={'color': "white"})
        st.plotly_chart(fig_oee, use_container_width=True, key=chart_key)

# Handle Automated Batch Simulation
if run_batch:
    batch_scenarios = [
        "Golden Sample (100% Healthy PCB)",
        "Inject: Solder Bridge Short (IC Pins)",
        "Golden Sample (100% Healthy PCB)",
        "Inject: MOSFET Thermal Overheat (>95°C)",
        "Golden Sample (100% Healthy PCB)"
    ]
    for idx, b_scen in enumerate(batch_scenarios):
        r_v, e_v, i_v, d_list, t_v = generate_and_inspect_pcb(b_scen, canny_low, canny_high, thermal_limit)
        p_flag = len(d_list) == 0
        st.session_state.total_inspected += 1
        if p_flag:
            st.session_state.pass_count += 1
        else:
            st.session_state.reject_count += 1
            for d_name, _, _ in d_list:
                st.session_state.defect_tally[d_name] = st.session_state.defect_tally.get(d_name, 0) + 1

        new_unit_id = f"PCB-0{st.session_state.total_inspected}"
        st.session_state.plc_logs.insert(0, {
            "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "Unit_ID": new_unit_id,
            "Result": "PASS" if p_flag else "REJECT",
            "Defect_Type": "None" if p_flag else d_list[0][0],
            "PLC_Action": "Q0.0 Conveyor Continue" if p_flag else "Q0.1 Pneumatic Ejector ON",
            "Confidence": "99.5%" if p_flag else f"{d_list[0][1]:.1f}%"
        })
        with viewport_placeholder.container():
            render_main_station(r_v, e_v, i_v, p_flag, d_list, t_v, chart_key=f"oee_batch_{idx}_{ time.time() }")
        time.sleep(0.55)
else:
    with viewport_placeholder.container():
        render_main_station(rgb_hud, edge_hud, ir_hud, is_pass, active_defects, max_ir_temp, chart_key="oee_default")

# ---------------------------------------------------------
# BOTTOM ANALYTICS: PARETO DEFECT CHART & PLC HISTORIAN LOG
# ---------------------------------------------------------
st.divider()
col_pareto, col_table = st.columns([1.4, 2.6])

with col_pareto:
    st.subheader("📊 Defect Pareto Distribution")
    df_defects = pd.DataFrame({
        "Defect Category": list(st.session_state.defect_tally.keys()),
        "Occurrences": list(st.session_state.defect_tally.values())
    }).sort_values(by="Occurrences", ascending=True)

    fig_bar = go.Figure(go.Bar(
        x=df_defects["Occurrences"],
        y=df_defects["Defect Category"],
        orientation='h',
        marker=dict(color='#00f2fe', line=dict(color='#00e676', width=1.5))
    ))
    fig_bar.update_layout(
        height=240, template="plotly_dark", paper_bgcolor="#080b11", plot_bgcolor="#101520",
        margin=dict(l=10, r=15, t=10, b=10), xaxis_title="Rejected Units Count"
    )
    st.plotly_chart(fig_bar, use_container_width=True, key="pareto_bar_chart")

with col_table:
    st.subheader("📋 PROFINET Inspection Log & Export")
    df_logs = pd.DataFrame(st.session_state.plc_logs[:8])
    st.dataframe(df_logs, use_container_width=True, hide_index=True)
    
    csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Vision & PLC Quality Report (.CSV)",
        data=csv_bytes,
        file_name="AOI_Vision_PLC_Inspection_Report.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------
# DEVELOPER CREDITS & FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; padding: 15px; background-color: #101520; border-radius: 8px; border: 1px solid #1e293b; margin-top: 15px;">
        <p style="color: #64748b; margin-bottom: 4px; font-size: 0.82rem; letter-spacing: 1.2px; text-transform: uppercase;">
            Industry 4.0 Machine Vision & PLC Digital Twin Architecture
        </p>
        <h4 style="color: #00f2fe; margin: 0; font-family: monospace; font-size: 1.15rem;">
            Designed & Developed by Amirhosein Razeghi
        </h4>
        <p style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">
            Electrical Engineering | K. N. Toosi University of Technology
        </p>
    </div>
""", unsafe_allow_html=True)