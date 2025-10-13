from Air import AirBubble, Agent
from Air import Tools
import json

llm_config = {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "api_key": "AIzaSyCH8LRvlaUti2aLuss1i-zqjcPyQej1Q1s",
    "temperature": 0.6,
    "max_tokens": 6000,
}

# ========================================================================
# TOOLS - Just inherit, set name/description, override run()
# ========================================================================


class PUBGWeaponInfo(Tools):
    def __init__(self):
        super().__init__(
            name="pubg_weapon_info",
            description="Get information about PUBG Mobile weapons and items including rarity and how to obtain them.",
        )

    def run(self, weapon_name: str) -> str:
        """Look up a weapon in the database."""
        database = {
            "karambit": {
                "name": "Karambit Knife",
                "type": "Melee",
                "rarity": "Legendary",
                "damage": 55,
                "how_to_get": ["Lucky Spin", "Premium BP shop", "Draw events"],
                "notes": "One of the rarest melee weapons",
            },
            "m24": {
                "name": "M24",
                "type": "Sniper",
                "rarity": "Common",
                "damage": 88,
                "how_to_get": ["Ground loot", "Airdrops"],
            },
            "ak47": {
                "name": "AK47",
                "type": "Assault Rifle",
                "rarity": "Common",
                "damage": 49,
                "how_to_get": ["Ground loot", "Crates"],
            },
        }

        weapon = database.get(weapon_name.lower())
        if weapon:
            return json.dumps({"success": True, "weapon": weapon})
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Weapon '{weapon_name}' not found",
                    "available": list(database.keys()),
                }
            )


class LuckySpinInfo(Tools):
    def __init__(self):
        super().__init__(
            name="lucky_spin_info",
            description="Get information about PUBG Mobile Lucky Spin system, rates, and strategies.",
        )

    def run(self, info_type: str) -> str:
        """Get info about lucky spin."""
        info = {
            "cost": "Each spin costs 10 UC (Unknown Cash)",
            "pity_system": "Guaranteed legendary after 10 spins (100 UC total)",
            "best_time": "Spin during new season launches for higher rates",
            "strategy": "Save UC over multiple seasons and use pity system",
            "tips": [
                "Check spin history for remaining pulls",
                "Legendary items rotate monthly",
                "First pull has boosted rates",
            ],
        }

        if info_type.lower() in info:
            return json.dumps({"success": True, "info": info[info_type.lower()]})
        else:
            return json.dumps(
                {
                    "success": True,
                    "all_info": info,
                }
            )


class PUBGEventInfo(Tools):
    def __init__(self):
        super().__init__(
            name="pubg_events",
            description="Check active PUBG Mobile events and special opportunities.",
        )

    def run(self, event_name: str = "") -> str:
        """Get event information."""
        events = {
            "survivor_pass": {
                "type": "Free reward track",
                "rewards": ["BP", "Cosmetics", "UC"],
                "duration": "Seasonal",
            },
            "royal_pass": {
                "type": "Premium battle pass",
                "cost": "UC",
                "rewards": ["Skins", "Weapons", "UC back"],
            },
            "lucky_spin": {
                "type": "Gacha system",
                "frequency": "Weekly rotations",
                "pity": "10 spins guaranteed legendary",
            },
        }

        if event_name:
            event = events.get(event_name.lower())
            if event:
                return json.dumps({"success": True, "event": event})
            else:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"Event '{event_name}' not found",
                        "available": list(events.keys()),
                    }
                )
        else:
            return json.dumps({"success": True, "events": list(events.keys())})


class CostAnalysis(Tools):
    def __init__(self):
        super().__init__(
            name="cost_analysis",
            description="Analyze costs and calculate UC needed for items or spins.",
        )

    def run(self, item: str, quantity: int = 1) -> str:
        """Calculate costs for items."""
        costs = {
            "spin": 10,
            "guaranteed_spin": 100,
            "royal_pass": 600,
        }

        if item.lower() in costs:
            unit_cost = costs[item.lower()]
            total = unit_cost * quantity
            return json.dumps(
                {
                    "success": True,
                    "item": item,
                    "quantity": quantity,
                    "unit_cost_uc": unit_cost,
                    "total_uc": total,
                }
            )
        else:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Item '{item}' not found",
                    "available_items": list(costs.keys()),
                }
            )


# ========================================================================
# TOOLS LIST
# ========================================================================

tools = [
    PUBGWeaponInfo(),
    LuckySpinInfo(),
    PUBGEventInfo(),
    CostAnalysis(),
]

# ========================================================================
# AGENTS WITH TOOLS
# ========================================================================

agents = [
    Agent(
        name="FactCheckAgent",
        role="Fact Checker",
        description="Verifies all information and assumptions to ensure accuracy.",
        goal="Ensure all data and claims are correct before execution.",
        tools=tools,
        llm_config=llm_config,
    ),
    Agent(
        name="ThinkTankAgent",
        role="Analytical Thinker",
        description="Breaks down situations logically and evaluates options.",
        goal="Provide structured insights and potential outcomes.",
        tools=tools,
        llm_config=llm_config,
    ),
    Agent(
        name="ReviewAgent",
        role="Reviewer",
        description="Critically reviews proposals and identifies weaknesses.",
        goal="Improve strategies by pointing out flaws and risks.",
        tools=tools,
        llm_config=llm_config,
    ),
    Agent(
        name="RiskAgent",
        role="Risk Assessor",
        description="Analyzes potential risks and suggests mitigation.",
        goal="Minimize failures and negative outcomes.",
        tools=tools,
        llm_config=llm_config,
    ),
    Agent(
        name="DataAgent",
        role="Data Analyst",
        description="Interprets and provides insights from available data.",
        goal="Use data-driven insights to guide decisions.",
        tools=tools,
        llm_config=llm_config,
    ),
    Agent(
        name="ImplementAgent",
        role="Execution Planner",
        description="Turns ideas into actionable step-by-step plans.",
        goal="Create a concrete plan to implement strategies.",
        tools=tools,
        llm_config=llm_config,
    ),
]

# ========================================================================
# RUN
# ========================================================================

result = AirBubble(
    name="Deep thinker",
    goal="HOW TO GET KARAMBIT KNIFE IN PUBG MOBILE",
    agents=agents,
    llm_config=llm_config,
    mode="discussion",
).run(user_input="")

print("\n" + "=" * 80)
print("FINAL RESULT:")
print("=" * 80)
print(result)
