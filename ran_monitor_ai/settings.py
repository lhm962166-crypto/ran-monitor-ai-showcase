"""Project constants and operating rules."""

KPIS = (
    "RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS", "UL_NPRB",
    "UL_SNR", "TX_Bytes", "RX_Bytes", "Estimated_UL_Buffer",
    "PRBs_DL_Current", "PRBs_UL_Current", "PRB_Utilization_DL",
    "PRB_Utilization_UL", "UL_NumberOfPackets", "DL_NumberOfPackets",
)
WINDOW_SIZE = 128
POLICY_ZH = {"balanced": "均衡", "sensitive": "高召回", "precision": "高精确"}
CAUSE_MAP = {
    "Jamming": "Interference",
    "Co-Channel Interference (Severe)": "Interference",
    "Co-Channel Interference (Mild)": "Interference",
    "High Network Congestion (Gradual Buildup)": "Congestion",
    "High Network Congestion (Sudden Spike)": "Congestion",
    "Resource Allocation Bugs": "Congestion",
    "Buffer Overflow (Gradual Buildup)": "Congestion",
    "Antenna Failure": "Equipment",
    "Faulty RF Filters (Temporal)": "Equipment",
    "Doppler Shift (Severe)": "Mobility",
    "Faulty Handover Algorithm (Too Frequent)": "Mobility",
}
CAUSE_ZH = {
    "Interference": "无线干扰",
    "Congestion": "网络拥塞",
    "Equipment": "设备故障",
    "Mobility": "移动性 / 切换问题",
    "Normal": "正常",
}
ACTION = {
    "Interference": ("检查频谱并定位干扰源", "无线工程团队"),
    "Congestion": ("检查 PRB 利用率，评估负载均衡或扩容", "容量规划团队"),
    "Equipment": ("检查天线、射频滤波器和基站设备状态", "现场运维团队"),
    "Mobility": ("检查切换参数、移动场景和多普勒影响", "无线优化团队"),
}
