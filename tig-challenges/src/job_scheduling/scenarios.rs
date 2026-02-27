pub struct ScenarioConfig {
    pub avg_op_flexibility: f32,
    pub reentrance_level: f32,
    pub flow_structure: f32,
    pub product_mix_ratio: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scenario {
    FLOW_SHOP,
    HYBRID_FLOW_SHOP,
    JOB_SHOP,
    FJSP_MEDIUM,
    FJSP_HIGH,
}

impl From<Scenario> for ScenarioConfig {
    fn from(scenario: Scenario) -> Self {
        match scenario {
            Scenario::FLOW_SHOP => ScenarioConfig {
                avg_op_flexibility: 1.0,
                reentrance_level: 0.2,
                flow_structure: 0.0,
                product_mix_ratio: 0.5,
            },
            Scenario::HYBRID_FLOW_SHOP => ScenarioConfig {
                avg_op_flexibility: 3.0,
                reentrance_level: 0.2,
                flow_structure: 0.0,
                product_mix_ratio: 0.5,
            },
            Scenario::JOB_SHOP => ScenarioConfig {
                avg_op_flexibility: 1.0,
                reentrance_level: 0.0,
                flow_structure: 0.4,
                product_mix_ratio: 1.0,
            },
            Scenario::FJSP_MEDIUM => ScenarioConfig {
                avg_op_flexibility: 3.0,
                reentrance_level: 0.2,
                flow_structure: 0.4,
                product_mix_ratio: 1.0,
            },
            Scenario::FJSP_HIGH => ScenarioConfig {
                avg_op_flexibility: 10.0,
                reentrance_level: 0.0,
                flow_structure: 1.0,
                product_mix_ratio: 1.0,
            },
        }
    }
}

impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scenario::FLOW_SHOP => write!(f, "flow_shop"),
            Scenario::HYBRID_FLOW_SHOP => write!(f, "hybrid_flow_shop"),
            Scenario::JOB_SHOP => write!(f, "job_shop"),
            Scenario::FJSP_MEDIUM => write!(f, "fjsp_medium"),
            Scenario::FJSP_HIGH => write!(f, "fjsp_high"),
        }
    }
}

impl std::str::FromStr for Scenario {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "flow_shop" => Ok(Scenario::FLOW_SHOP),
            "hybrid_flow_shop" => Ok(Scenario::HYBRID_FLOW_SHOP),
            "job_shop" => Ok(Scenario::JOB_SHOP),
            "fjsp_medium" => Ok(Scenario::FJSP_MEDIUM),
            "fjsp_high" => Ok(Scenario::FJSP_HIGH),
            _ => Err(anyhow::anyhow!("Invalid scenario type: {}", s)),
        }
    }
}
