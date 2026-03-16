use super::types::Pre;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectedTrack {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

pub fn detect_track(pre: &Pre) -> DetectedTrack {
    if pre.chaotic_like {
        DetectedTrack::FjspHigh
    } else if pre.flow_like > 0.82 && pre.jobshopness < 0.38 && pre.high_flex < 0.3 {
        DetectedTrack::FlowShop
    } else if pre.flow_like > 0.45 && pre.jobshopness < 0.55 && pre.flex_avg > 2.0 && pre.flex_avg < 4.0 && pre.high_flex < 0.1 {
        DetectedTrack::HybridFlowShop
    } else if pre.jobshopness > 0.5 && pre.high_flex < 0.3 && pre.flow_like > 0.35 {
        DetectedTrack::JobShop
    } else if pre.high_flex > 0.4 && pre.jobshopness > 0.4 {
        DetectedTrack::FjspMedium
    } else {
        DetectedTrack::FjspMedium
    }
}
