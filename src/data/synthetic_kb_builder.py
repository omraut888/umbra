"""Build a synthetic knowledge base with known coverage ground truth.

The KB is a home food-gardening knowledge base. Six *evaluated* topics are
baked in at three coverage tiers:

    FULL    composting, tomato_growing        (9 dedicated documents each)
    THIN    hydroponics, mushroom_cultivation (a 2-3 sentence mention buried
                                               in one document about something else)
    ABSENT  cryptocurrency_taxation,          (zero documents; clearly outside
            orbital_mechanics                  the gardening domain)

The remaining documents cover *background* gardening topics (soil, irrigation,
pests, ...). They make the KB realistic: thin-topic mentions have to be buried
somewhere, and absent topics must score low against a broad KB rather than an
empty one. Background topics are recorded in ground_truth.json but are not
part of the tiered evaluation.

Documents are hand-written rather than LLM-generated so that the ground truth
is exact: no evaluated thin/absent topic is mentioned anywhere except the
designated passages. `verify_no_leakage` enforces this at build time.

Output layout (default: data/synthetic_kb/):
    docs/<doc_id>.md     one markdown file per document (what the RAG system indexes)
    ground_truth.json    topic -> tier mapping, plus document labels

Usage:
    python -m src.data.synthetic_kb_builder [--out data/synthetic_kb]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DEFAULT_OUT = Path("data/synthetic_kb")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    topic: str
    body: str

    def to_markdown(self) -> str:
        return f"# {self.title}\n\n{self.body.strip()}\n"


EVALUATED_TOPICS: Dict[str, Dict[str, str]] = {
    "composting": {
        "tier": "full",
        "description": (
            "Home composting: building and managing compost piles and bins, the "
            "carbon-to-nitrogen ratio of browns and greens, moisture and aeration, "
            "hot versus cold composting temperatures, vermicomposting with worms, "
            "troubleshooting smelly or slow piles, and using finished compost."
        ),
    },
    "tomato_growing": {
        "tier": "full",
        "description": (
            "Growing tomatoes in a home garden: choosing determinate versus "
            "indeterminate varieties, starting and transplanting seedlings, staking "
            "and caging, pruning suckers, watering and fertilizing, diseases such as "
            "early blight and blossom end rot, pests such as hornworms, and harvesting."
        ),
    },
    "hydroponics": {
        "tier": "thin",
        "description": (
            "Hydroponics: growing plants without soil in a nutrient solution, "
            "including deep water culture, nutrient film technique, ebb and flow and "
            "Kratky systems, mixing nutrients, managing solution pH and electrical "
            "conductivity (EC), lighting, and suitable crops."
        ),
    },
    "mushroom_cultivation": {
        "tier": "thin",
        "description": (
            "Growing edible mushrooms at home: oyster, shiitake and wine cap "
            "mushrooms, spawn types, inoculating logs, straw or wood chips, "
            "pasteurization and sterilization of substrate, humidity and fruiting "
            "conditions, contamination, and harvesting flushes."
        ),
    },
    "cryptocurrency_taxation": {
        "tier": "absent",
        "description": (
            "Tax treatment of cryptocurrency: capital gains on selling or trading "
            "Bitcoin and other tokens, cost basis methods, taxable events, staking and "
            "mining income, reporting forms, wash sales, and record keeping for "
            "digital asset transactions."
        ),
    },
    "orbital_mechanics": {
        "tier": "absent",
        "description": (
            "Orbital mechanics and spaceflight: Kepler's laws, orbital velocity and "
            "escape velocity, Hohmann transfer orbits, delta-v budgets, geostationary "
            "and low Earth orbits, orbital inclination changes, and gravity assists."
        ),
    },
}

# Terms that must not appear in the KB except in the designated thin passages.
# Absent-topic terms must never appear at all.
LEAKAGE_TERMS: Dict[str, List[str]] = {
    "hydroponics": ["hydropon", "nutrient solution", "kratky", "deep water culture", "nutrient film"],
    "mushroom_cultivation": ["mushroom", "spawn", "mycel", "stropharia", "shiitake", "oyster"],
    "cryptocurrency_taxation": ["crypto", "bitcoin", "capital gain", "blockchain", "token"],
    "orbital_mechanics": ["orbit", "satellite", "spacecraft", "delta-v", "kepler", "rocket"],
}

THIN_PASSAGES: Dict[str, str] = {
    "hydroponics": (
        "Some growers skip soil entirely and raise lettuce and basil hydroponically, "
        "with the roots suspended in a nutrient solution instead of potting mix. The "
        "simplest version, the Kratky method, is just a jar of nutrient solution and a "
        "net cup, with no pump. It is a niche approach, and the rest of this guide "
        "assumes an ordinary seed-starting mix."
    ),
    "mushroom_cultivation": (
        "A wood chip path can double as a small mushroom bed: wine cap mushroom spawn "
        "mixed into fresh hardwood chips in spring can fruit by late summer or the "
        "following year. Keep the chips moist and partly shaded if you try it."
    ),
}


COMPOSTING_DOCS = [
    Document("compost-01-basics", "Composting Basics: Browns, Greens, Air and Water", "composting", """
Compost is organic matter that microbes have broken down into a dark, crumbly material that feeds soil life and improves soil structure. A home pile needs four ingredients: carbon-rich "browns", nitrogen-rich "greens", air, and water.

Browns include dry leaves, straw, shredded cardboard, newspaper and wood chips. Greens include vegetable scraps, coffee grounds, fresh grass clippings, and manure from herbivores such as chickens, horses and rabbits. Browns supply energy for microbes; greens supply the nitrogen they need to build protein.

A good working rule is roughly three parts browns to one part greens by volume. Too many greens and the pile turns slimy and smells of ammonia; too many browns and it sits for months without heating up.

Keep the pile about as damp as a wrung-out sponge and turn it every week or two to add oxygen. A well-managed pile produces finished compost in two to four months; a neglected one still breaks down, but it can take a year or more.
"""),
    Document("compost-02-cn-ratio", "Understanding the Carbon-to-Nitrogen Ratio in Compost", "composting", """
Composting microbes work fastest when the pile's overall carbon-to-nitrogen (C:N) ratio sits between about 25:1 and 30:1. Every material has its own ratio, so the goal is to blend ingredients until the mix lands in that range.

Typical values: fresh grass clippings are around 20:1, vegetable scraps 15:1 to 20:1, coffee grounds about 20:1, and poultry manure can be as low as 7:1. On the carbon side, dry autumn leaves range from 40:1 to 80:1, straw is roughly 80:1, cardboard about 350:1, and sawdust can exceed 400:1.

When the ratio is too low (excess nitrogen), microbes cannot use all the nitrogen and the surplus escapes as ammonia gas, which is where the sharp smell comes from. When the ratio is too high (excess carbon), microbes run short of nitrogen, growth slows, and decomposition crawls.

You don't need to calculate precisely. If the pile smells, add browns. If it stays cold and dry-looking, add greens and water. Mixing materials thoroughly matters as much as the proportions.
"""),
    Document("compost-03-hot-composting", "Hot Composting and Thermophilic Temperatures", "composting", """
Hot composting deliberately drives a pile into the thermophilic range, roughly 131 to 160 degrees Fahrenheit (55 to 71 degrees Celsius). At these temperatures, heat-loving bacteria break material down quickly, and most weed seeds and plant pathogens are destroyed.

To get a pile hot you need mass: at least one cubic yard (about three feet on each side) so the core stays insulated. Build it all at once rather than adding a little at a time, keep the carbon-to-nitrogen ratio near 30:1, and keep it evenly moist.

A compost thermometer with a long probe is the easiest way to track progress. A typical hot pile peaks within a few days, holds high temperatures for a week or two, then cools as the easy food runs out. Turning the pile moves the cooler outer material into the center and brings on another heating cycle.

To reliably kill weed seeds, the whole pile should spend at least three days above 131°F, which usually takes three to five turns so that every part passes through the hot core. Temperatures above 160°F are counterproductive: they kill the beneficial microbes too, so turn the pile or add browns if it overheats.
"""),
    Document("compost-04-cold-composting", "Cold Composting: The Low-Effort Method", "composting", """
Cold composting, sometimes called passive composting, means adding material to a pile or bin as it becomes available and largely leaving it alone. The pile never gets very hot, so it relies on slower mesophilic microbes, fungi, worms and insects to do the work.

The main advantage is effort: there is no need to gather a large batch of material at once or turn the pile often. The trade-off is time. A cold pile typically takes six months to two years to finish, and because it never reaches thermophilic temperatures, weed seeds and diseased plant material can survive.

For that reason, keep weeds that have gone to seed, diseased tomato vines and invasive roots out of a cold pile. Chopping material into small pieces and keeping the pile moist will still speed things up considerably.

Many gardeners use a two-bin system for cold composting: one bin receives fresh material while the other, full bin is left to finish. When the second bin is ready, harvest it from the bottom and swap roles.
"""),
    Document("compost-05-vermicomposting", "Vermicomposting with Red Wiggler Worms", "composting", """
Vermicomposting uses worms to turn food scraps into castings, an exceptionally rich form of compost. The right species is the red wiggler (Eisenia fetida), a surface-dwelling worm that thrives in decaying organic matter. Ordinary earthworms dug from the garden will not survive in a bin.

A worm bin can be a plastic tote with air holes drilled in the lid and sides. Fill it with moist bedding such as shredded newspaper, cardboard or coconut coir, add a handful of soil for grit, and introduce about one pound of worms. A pound of red wigglers can process roughly half a pound of food scraps a day once established.

Feed fruit and vegetable scraps, coffee grounds and crushed eggshells. Avoid meat, dairy, oily foods, and large amounts of citrus or onion, which can make the bin acidic or smelly. Bury food under the bedding to deter fruit flies.

Keep the bin between 55 and 77 degrees Fahrenheit and as damp as a wrung-out sponge. To harvest castings, push the contents to one side and put fresh bedding and food on the other; within a few weeks the worms migrate and the finished castings can be scooped out.
"""),
    Document("compost-06-troubleshooting", "Troubleshooting Common Compost Problems", "composting", """
A compost pile that smells like rotten eggs is anaerobic: it has too much water and too little air, and bacteria that don't need oxygen are producing hydrogen sulfide. Turn the pile to aerate it and mix in dry browns such as shredded leaves or straw to soak up excess moisture.

An ammonia smell means too much nitrogen. Mix in carbon-rich browns and turn the pile. This often happens after a large load of fresh grass clippings.

A pile that won't heat up is usually too small, too dry, or short on nitrogen. Check moisture first by squeezing a handful: it should feel like a wrung-out sponge. If it is moist but still cold, add greens such as fresh clippings, coffee grounds or manure.

Flies, rodents and raccoons are attracted to exposed food scraps, meat and dairy. Keep meat, bones and dairy out of the pile, bury kitchen scraps under at least eight inches of browns, and consider an enclosed bin with a lid and hardware cloth under the base.

If the pile is dry and dusty in the center, water it as you turn it, layer by layer, until everything is evenly moist.
"""),
    Document("compost-07-bins-tumblers", "Choosing a Compost Bin, Tumbler or Open Pile", "composting", """
Open piles are free and can be as large as you like, which makes them the easiest way to reach hot-composting temperatures. They are best suited to large gardens with plenty of yard waste and space away from the house.

Enclosed bins made from wood pallets, wire mesh or plastic keep the pile tidy, retain moisture and heat, and deter animals. A three-bin system lets you move material from one bin to the next as it matures: fresh material, active composting, and curing.

Compost tumblers are sealed barrels mounted on a frame so the contents can be turned with a crank. They make turning easy and keep pests out, but their small volume means they rarely reach high temperatures, and they must be filled in batches to be efficient. Dual-chamber tumblers let one side finish while the other is being filled.

Whatever you choose, site it on bare soil if possible so worms and microbes can move in, keep it within reach of a hose, and put it somewhere convenient enough that you will actually use it.
"""),
    Document("compost-08-using-finished", "Using Finished Compost in the Garden", "composting", """
Compost is finished when it is dark brown, crumbly, smells earthy, and you can no longer recognize the original ingredients. Let it cure for a few more weeks after it stops heating so that any remaining decomposition doesn't compete with plants for nitrogen.

Screening through a half-inch mesh removes sticks and unfinished chunks, which can go back into the next pile. Screened compost is ideal for potting mixes and top-dressing lawns.

In vegetable beds, spread one to three inches of compost each spring and either work it into the top few inches of soil or leave it on the surface as a mulch for worms to pull down. Side-dress heavy feeders mid-season with a handful around each plant.

Compost tea is made by steeping finished compost in water for a day or two and applying the liquid to soil. Evidence for dramatic benefits is mixed, but it is a gentle way to add soluble nutrients and microbes.

Compost is a soil conditioner more than a fertilizer. It typically contains only about one to two percent nitrogen, so very hungry crops may still need additional fertilizer.
"""),
    Document("compost-09-materials", "What You Can and Can't Compost", "composting", """
Most plant material can be composted. Good additions include fruit and vegetable scraps, coffee grounds and paper filters, tea leaves, eggshells, grass clippings, leaves, straw, shredded paper and cardboard, sawdust from untreated wood, and manure from herbivores.

Some materials are better left out of a home pile. Meat, fish, bones and dairy attract pests and can smell as they rot. Oils and fats slow decomposition. Pet waste from dogs and cats can carry pathogens that a home pile may not destroy. Diseased plants and weeds with mature seeds should only go into a hot pile that reliably exceeds 131°F.

Treat grass clippings from lawns sprayed with persistent herbicides with caution. Some broadleaf herbicides, such as aminopyralid, survive composting and can damage tomatoes, beans and peas when the compost is applied.

Glossy magazine paper, coal ash, and sawdust from pressure-treated lumber should be avoided. Black walnut leaves and hulls contain juglone, a compound that inhibits many plants; they break down eventually but are best composted separately and for longer.
"""),
]

TOMATO_DOCS = [
    Document("tomato-01-varieties", "Choosing Tomato Varieties: Determinate vs Indeterminate", "tomato_growing", """
Tomato varieties fall into two growth habits. Determinate, or bush, tomatoes grow to a fixed height of about three to four feet, set most of their fruit over two or three weeks, and then decline. They suit containers, small gardens, and anyone planning to can a large harvest at once. 'Roma', 'Celebrity' and 'Rutgers' are common determinate varieties.

Indeterminate tomatoes keep growing and producing until frost kills them, often reaching six to ten feet. They need tall stakes, cages or trellises but provide a steady harvest all season. 'Sungold', 'Brandywine', 'Cherokee Purple' and 'Better Boy' are popular indeterminate types.

Beyond habit, consider days to maturity (early varieties ripen in about 55 to 65 days from transplant, main-season in 70 to 80), fruit type (cherry, paste, slicer or beefsteak), and disease resistance. Seed packets list resistance codes: V for verticillium wilt, F for fusarium wilt, N for nematodes, and T for tobacco mosaic virus.

Heirlooms are open-pollinated varieties passed down for generations and prized for flavor. Hybrids, marked F1, often add vigor and disease resistance, but saved seed will not grow true to type.
"""),
    Document("tomato-02-starting-seeds", "Starting Tomato Seeds Indoors", "tomato_growing", """
Start tomato seeds indoors six to eight weeks before your average last frost date. Sow them a quarter inch deep in a sterile seed-starting mix and keep the soil between 70 and 80 degrees Fahrenheit; a heat mat speeds germination, which usually takes five to ten days.

As soon as seedlings emerge, give them strong light for 14 to 16 hours a day. A windowsill is rarely bright enough and produces tall, spindly plants, so most growers use shop lights or LED grow lights hung two to four inches above the seedlings and raised as the plants grow.

When seedlings develop their first true leaves, thin them to one per cell or pot them up into three- to four-inch containers. Tomatoes are one of the few vegetables that benefit from being buried deeper at each potting-up, because they grow roots all along the buried stem.

Water from the bottom when the surface feels dry, and start feeding with a half-strength liquid fertilizer once true leaves appear. A small fan blowing gently across the seedlings, or brushing them with your hand daily, produces sturdier stems.
"""),
    Document("tomato-03-transplanting", "Hardening Off and Transplanting Tomatoes", "tomato_growing", """
Tomatoes are tender and should not go into the garden until all danger of frost has passed and nighttime temperatures stay above 50 degrees Fahrenheit. Cold soil stunts them, so waiting a week or two past the last frost date often gives better results than planting early.

Harden seedlings off over seven to ten days before transplanting. Start with an hour or two outdoors in a sheltered, shaded spot, and gradually increase their time outside and their exposure to direct sun and wind. Skipping this step can cause sunscald and transplant shock.

Choose a site with at least six to eight hours of direct sun. Space determinate plants about two feet apart and indeterminate plants two and a half to three feet apart, with three to four feet between rows, so air circulates and foliage dries quickly.

Plant deeply: strip the lower leaves and bury two-thirds of the stem, either in a deep hole or laid sideways in a trench. The buried stem will sprout roots, producing a stronger plant. Water in thoroughly and install stakes or cages at planting time to avoid damaging roots later.
"""),
    Document("tomato-04-staking-pruning", "Staking, Caging and Pruning Tomato Plants", "tomato_growing", """
Supporting tomatoes keeps fruit off the ground, improves airflow and makes harvesting easier. Common methods are single stakes, wire cages, the Florida weave (twine woven between stakes along a row), and string trellises hanging from an overhead support.

Cages suit determinate varieties and require little maintenance. Indeterminate varieties outgrow most store-bought cages; sturdy homemade cages of concrete reinforcing mesh or tall stakes work better.

Suckers are the shoots that form in the joint between the main stem and a leaf branch. On indeterminate tomatoes, removing suckers channels energy into fewer, larger fruits and keeps the plant manageable, especially when growing on a single stake. Pinch them off when they are two to four inches long. Many growers keep one or two suckers low on the plant as additional leaders.

Do not prune determinate tomatoes heavily; because their fruit sets on a fixed number of branches, removing suckers reduces the harvest.

Removing the lowest leaves, up to about twelve inches off the ground, reduces soil splash that spreads early blight and septoria leaf spot.
"""),
    Document("tomato-05-water-feed", "Watering and Fertilizing Tomatoes", "tomato_growing", """
Tomatoes need about one to two inches of water per week, more in hot, windy weather and when fruit is sizing up. Deep, infrequent watering encourages deep roots; a thorough soaking two or three times a week is better than a light daily sprinkle.

Consistency matters as much as quantity. Swinging between dry and soaked soil contributes to blossom end rot and fruit cracking. A two- to three-inch layer of straw or shredded leaves helps hold soil moisture steady. Water at the base of the plant, ideally in the morning, and keep the foliage dry to reduce disease.

Before planting, work compost and a balanced fertilizer into the bed. Too much nitrogen produces lush foliage with few flowers, so once plants begin to set fruit, switch to a fertilizer with less nitrogen relative to phosphorus and potassium, or side-dress with compost every three to four weeks.

Container tomatoes dry out quickly and may need watering daily in summer; use at least a five-gallon pot, and preferably ten gallons or more for indeterminate varieties.
"""),
    Document("tomato-06-diseases", "Common Tomato Diseases and How to Manage Them", "tomato_growing", """
Early blight, caused by the fungus Alternaria, produces brown spots with concentric rings, like a target, on lower leaves, which then yellow and drop. Septoria leaf spot causes many small circular spots with dark borders and gray centers. Both spread from soil splash, so mulching, removing lower leaves, and rotating crops are the best defenses.

Late blight is more destructive: large, greasy-looking gray-green lesions on leaves and firm brown patches on fruit, often with white fuzzy growth in humid weather. It can wipe out plants within days. Remove and bag infected plants rather than composting them.

Fusarium and verticillium wilts are soilborne fungi that block the plant's water-conducting tissue, causing yellowing and wilting that often starts on one side of the plant. There is no cure; plant resistant varieties (marked F and V) and rotate tomatoes out of affected beds for several years.

Blossom end rot, a dark, sunken, leathery patch on the bottom of the fruit, is not a disease but a calcium disorder usually caused by uneven watering rather than a lack of calcium in the soil. Steady moisture and mulch generally resolve it.
"""),
    Document("tomato-07-pests", "Tomato Pests: Hornworms, Aphids and More", "tomato_growing", """
Tomato hornworms are large green caterpillars, up to four inches long, with a horn on the rear end. They can strip a plant of leaves in a day or two. Hand-pick them in the early morning or at dusk. Hornworms covered in small white cocoons have been parasitized by braconid wasps; leave them in place so the next generation of wasps emerges.

Aphids cluster on new growth and the undersides of leaves. A strong spray of water knocks them off, and ladybugs and lacewings usually bring them under control. Insecticidal soap works for heavy infestations.

Whiteflies and spider mites are more common in hot, dry conditions and in greenhouses. Spider mites cause fine stippling on leaves and webbing; spraying the undersides of leaves with water and keeping plants well watered helps.

Stink bugs pierce fruit and leave cloudy, pithy spots under the skin. Row cover early in the season and hand-picking reduce damage. Birds and squirrels also peck ripening fruit; picking at the first blush of color and ripening indoors avoids losses.
"""),
    Document("tomato-08-harvest", "Harvesting and Ripening Tomatoes", "tomato_growing", """
Tomatoes are ready to pick when they have reached full color for the variety and give slightly when squeezed. Fruit picked at the "breaker" stage, when color first starts to blush, will ripen fully indoors at room temperature with essentially no loss of flavor, which also protects it from cracking, pests and birds.

Ripening is driven by temperature more than sunlight. Above about 85 to 90 degrees Fahrenheit, tomatoes stop producing the red pigment lycopene and may stay orange-yellow, so in very hot weather it is better to pick at the breaker stage and ripen them indoors.

Never refrigerate ripe tomatoes if you can avoid it; temperatures below 55°F damage flavor and texture. Store them stem side down at room temperature, out of direct sun.

At the end of the season, before the first frost, harvest all mature green tomatoes. Wrap them individually in newspaper or place them in a box with a banana, whose ethylene gas speeds ripening, and check them every few days. Alternatively, pull whole plants and hang them upside down in a garage.
"""),
    Document("tomato-09-cracking-problems", "Why Tomatoes Crack, Split or Fail to Set Fruit", "tomato_growing", """
Cracking occurs when a tomato's interior grows faster than its skin can stretch, usually after heavy rain or watering following a dry spell. Radial cracks run from the stem outward; concentric cracks form rings around the stem. Consistent watering and mulch reduce cracking, and crack-resistant varieties are available.

Catfacing is puckered, scarred tissue at the blossom end of the fruit. It is caused by cool temperatures during flowering disturbing fruit development and is most common in large beefsteak types.

Blossom drop, where flowers fall off without setting fruit, usually happens when daytime temperatures exceed about 90°F or nights stay above 75°F, or when nights fall below 55°F. Pollen becomes sterile at these extremes. Fruit set generally resumes when temperatures moderate. Gently shaking the plants or tapping flower clusters during the day helps release pollen.

Sunscald appears as pale, papery patches on fruit exposed to intense sun, often after heavy pruning or leaf loss from disease. Keep enough foliage to shade developing fruit, and use shade cloth during heat waves.
"""),
]

BACKGROUND_DOCS = [
    Document("soil-01-testing", "Soil Testing: What to Test and Why", "soil", """
A soil test is the most reliable way to know what your garden actually needs. A standard test from a cooperative extension lab reports pH, phosphorus, potassium, calcium, magnesium and organic matter, and gives fertilizer recommendations for the crops you plan to grow.

Take samples from six to ten spots across a bed, digging down about six inches, and mix them together in a clean bucket. Send about two cups of the combined sample. Test in autumn so there is time to make adjustments before spring planting, and retest every two to three years.

Home test kits give a rough idea of pH and major nutrients, but lab tests are far more accurate and usually inexpensive.

If you garden in an urban area, ask the lab for a lead test as well. Lead from old paint and past vehicle exhaust can accumulate in soil near buildings and roads; raised beds with imported soil are a common solution where levels are high.
"""),
    Document("soil-02-ph", "Adjusting Soil pH with Lime and Sulfur", "soil", """
Soil pH controls how available nutrients are to plant roots. Most vegetables grow best in slightly acidic soil, between about 6.0 and 7.0. Blueberries prefer strongly acidic soil around 4.5 to 5.5, while brassicas tolerate slightly alkaline conditions.

To raise pH in acidic soil, apply garden lime. Calcitic lime adds calcium; dolomitic lime adds both calcium and magnesium and is the better choice if a soil test shows low magnesium. Lime works slowly, taking several months to change pH, so apply it in autumn.

To lower pH, apply elemental sulfur, which soil bacteria gradually convert to sulfuric acid. Like lime, it acts over months, and the amount needed depends on soil texture: clay soils require more than sandy soils to shift the same amount.

Change pH gradually and base the amount on a soil test. Over-liming is hard to reverse and can lock up iron, manganese and phosphorus.
"""),
    Document("soil-03-structure", "Improving Clay and Sandy Soils", "soil", """
Soil texture is determined by its proportions of sand, silt and clay. Clay soils hold water and nutrients well but drain slowly, compact easily and warm up late in spring. Sandy soils drain quickly and warm early but hold little water or nutrients.

The remedy for both is the same: add organic matter. In clay, organic matter binds fine particles into larger crumbs, creating pore spaces for air and drainage. In sand, it acts like a sponge that holds water and nutrients.

Avoid working clay soil when it is wet; squeeze a handful, and if it forms a sticky ribbon, wait. Tilling wet clay destroys structure and creates hard clods. Adding sand to clay in small amounts can actually make it worse, producing something close to concrete.

Reduce compaction by establishing permanent beds and paths, so you never walk where plants grow, and by keeping the soil covered with mulch or cover crops.
"""),
    Document("water-01-drip", "Drip Irrigation for Vegetable Gardens", "irrigation", """
Drip irrigation delivers water directly to the soil at the base of plants through tubing and emitters. It uses 30 to 50 percent less water than sprinklers, keeps foliage dry, which reduces fungal disease, and doesn't water the paths where weeds grow.

A basic system has a backflow preventer, a pressure regulator and a filter at the spigot, feeding half-inch mainline tubing. From the mainline, either run drip tape or emitter line with built-in emitters along each row, or punch individual emitters in at each plant. Emitters are rated in gallons per hour, commonly 0.5, 1 or 2 GPH.

Add a simple battery-powered timer and the system runs itself. Run it long enough to soak the root zone rather than frequently for short bursts; check by digging a few inches down an hour after watering.

Flush the lines at the start of each season, check emitters for clogs, and drain the system before the ground freezes.
"""),
    Document("water-02-when-how", "When and How Much to Water", "irrigation", """
Most vegetable gardens need about one inch of water per week from rain or irrigation, more in sandy soil and hot weather. A rain gauge in the garden tells you how much nature has already provided.

Water deeply and less often. Shallow daily watering keeps roots near the surface, where they dry out quickly. A deep soaking once or twice a week encourages roots to grow down to where the soil stays moist.

The best time to water is early morning. Less water is lost to evaporation, and any wet foliage dries quickly as the day warms, which limits disease. Evening watering leaves leaves damp overnight.

Check soil moisture with your finger: if the soil is dry two inches down, it is time to water. Seeds and new transplants are the exception and need the surface kept consistently moist until they are established.
"""),
    Document("beds-01-raised", "Building Raised Garden Beds", "raised_beds", """
Raised beds warm up faster in spring, drain well, and let you garden in places with poor or contaminated native soil. They also reduce bending and make it easy to keep foot traffic off growing areas.

A practical size is four feet wide, so you can reach the center from either side, and eight feet long. A depth of 10 to 12 inches suits most vegetables; carrots and parsnips appreciate more. Untreated cedar and redwood resist rot naturally and last ten years or more. Galvanized steel beds are durable and increasingly popular. Avoid old railroad ties.

Fill beds with a blend of roughly 60 percent topsoil and 40 percent compost, or buy a raised-bed mix from a landscape supplier. Bagged potting mix is too light and expensive for large beds.

Leave at least 18 inches, and preferably two feet, between beds for paths wide enough for a wheelbarrow. Lining the paths with cardboard and wood chips keeps weeds down.
"""),
    Document("beds-02-square-foot", "Square Foot Gardening and Intensive Spacing", "raised_beds", """
Square foot gardening divides a raised bed into a grid of one-foot squares, each planted with a set number of plants depending on their size. Large plants such as peppers, broccoli and cabbage get one square each; lettuce four per square; beets and spinach nine; carrots and radishes sixteen.

The method maximizes yield from a small area and makes planning straightforward. It works best with a rich, loose growing mix, because roots have less room to spread than in widely spaced rows.

Intensive spacing also shades the soil, which reduces weeds and moisture loss. The trade-off is reduced airflow, so disease-prone crops benefit from slightly wider spacing.

Vining crops such as cucumbers, peas and pole beans can be grown vertically on a trellis along the north side of the bed, where they won't shade shorter crops.
"""),
    Document("pest-01-ipm", "Integrated Pest Management in the Home Garden", "pest_management", """
Integrated pest management (IPM) is a decision-making approach that uses the least disruptive method that works. It starts with prevention: healthy soil, resistant varieties, crop rotation, proper spacing, and planting at the right time.

The next step is monitoring. Walk the garden regularly and inspect the undersides of leaves. Yellow sticky cards reveal flying pests such as whiteflies and fungus gnats. Most gardens can tolerate some damage; act only when pests reach a level that threatens the harvest.

When action is needed, start with physical controls: hand-picking, water sprays, row covers and barriers. Encourage beneficial insects by planting flowers such as dill, yarrow and alyssum.

Pesticides are the last resort. Choose the most targeted option, such as insecticidal soap for soft-bodied insects or Bacillus thuringiensis (Bt) for caterpillars, and apply it in the evening when pollinators are less active.
"""),
    Document("pest-02-slugs-rabbits", "Controlling Slugs, Snails, Rabbits and Deer", "pest_management", """
Slugs and snails chew ragged holes in leaves, mainly at night and in damp weather. Hand-pick them after dark with a flashlight, set boards on the ground and scrape off the slugs that hide beneath them in the morning, or use iron phosphate baits, which are safe around pets and wildlife. Copper tape around bed edges deters them.

Rabbits clip plants cleanly at an angle. A fence of chicken wire at least two feet high, with the bottom six inches buried or bent outward along the ground, keeps them out.

Deer are harder to exclude. They can jump fences lower than eight feet, although two shorter parallel fences set a few feet apart, or a slanted fence, can work because deer avoid jumping into confined spaces. Repellent sprays need reapplying after rain and work best when rotated.

Netting over brassicas and berries protects them from birds, and floating row cover protects young transplants from many insect pests.
"""),
    Document("seeds-01-indoor", "Seed Starting Indoors: Light, Heat and Containers", "seed_starting", """
Starting seeds indoors gives you a head start on the season and access to far more varieties than garden centers offer. You need containers with drainage, a sterile seed-starting mix, warmth, and above all strong light.

Seed-starting mix is fine-textured, sterile and low in nutrients, which reduces damping-off, a fungal disease that makes seedlings collapse at the soil line. Don't use garden soil indoors. Cell trays, soil blocks and recycled containers all work as long as they drain.

Most seeds germinate fastest at 70 to 80 degrees Fahrenheit, and a seedling heat mat under the tray provides steady bottom heat. Remove humidity domes as soon as seedlings emerge to improve airflow.

Light is where most indoor seedlings fail. Hang LED or fluorescent shop lights two to four inches above the plants and run them 14 to 16 hours a day on a timer. Some growers skip soil entirely and raise lettuce and basil hydroponically, with the roots suspended in a nutrient solution instead of potting mix. The simplest version, the Kratky method, is just a jar of nutrient solution and a net cup, with no pump. It is a niche approach, and the rest of this guide assumes an ordinary seed-starting mix.

Once seedlings have their first true leaves, begin feeding with a diluted fertilizer and pot them up as roots fill their cells.
"""),
    Document("seeds-02-direct-sowing", "Direct Sowing Seeds in the Garden", "seed_starting", """
Many crops grow best when sown directly where they will mature. Root crops such as carrots, radishes, beets and parsnips resent transplanting, and fast-growing crops such as beans, peas, squash, corn and cucumbers establish quickly from seed.

Check the seed packet for planting depth; a good rule is two to three times the seed's diameter. Tiny seeds such as carrots and lettuce need only a light covering or none at all.

Soil temperature determines germination speed. Peas germinate in soil as cool as 40°F, while beans need at least 60°F and squash and melons 65 to 70°F. A soil thermometer takes the guesswork out of timing.

Keep the seedbed consistently moist until seedlings emerge. A board or burlap laid over slow-germinating carrot seed holds in moisture; remove it as soon as the first sprouts appear. Thin seedlings to their final spacing once they have a set of true leaves.
"""),
    Document("mulch-01-guide", "A Practical Guide to Garden Mulch", "mulch", """
Mulch is any material spread over the soil surface. It suppresses weeds, holds moisture, moderates soil temperature, and as organic mulches break down, they feed soil life.

Straw is a favorite for vegetable beds: it is light, easy to spread and breaks down within a season. Make sure it is straw and not hay, which carries weed seeds. Shredded leaves are free and excellent. Grass clippings work if applied in thin layers so they don't mat and turn slimy.

Wood chips are best for paths and perennial plantings. They last several years and build fungal-rich soil. Contrary to a common myth, wood chips on the surface don't rob much nitrogen from the soil below; the effect is limited to the thin layer where chips touch the soil. A wood chip path can double as a small mushroom bed: wine cap mushroom spawn mixed into fresh hardwood chips in spring can fruit by late summer or the following year. Keep the chips moist and partly shaded if you try it.

Apply two to three inches of mulch after the soil has warmed in spring, keeping it an inch or two away from plant stems to prevent rot. In autumn, a thicker layer protects the roots of perennials and garlic over winter.
"""),
    Document("fruit-01-pruning", "Pruning Fruit Trees", "fruit_trees", """
Most fruit trees are pruned in late winter, while dormant and before buds break. Dormant pruning stimulates vigorous growth in spring. Summer pruning, by contrast, slows growth and is used to control the size of trees that are already vigorous.

Start by removing the three Ds: dead, damaged and diseased wood. Then remove water sprouts, the vertical shoots growing from branches, and suckers from the rootstock. Finally, thin crossing and inward-growing branches to open the canopy to light and air.

Apples and pears are usually trained to a central leader, a single main trunk with tiers of scaffold branches. Peaches and plums are trained to an open center, or vase shape, with three or four main scaffolds and no central leader.

Make each cut just outside the branch collar, the slightly swollen ring where a branch meets the trunk, and don't leave stubs or apply pruning paint; trees seal wounds best on their own.
"""),
    Document("fruit-02-thinning", "Thinning Fruit and Pollination for Home Orchards", "fruit_trees", """
Many fruit trees set more fruit than they can size up. Thinning, removing some young fruit a few weeks after bloom, produces larger, better fruit and prevents biennial bearing, where a tree crops heavily one year and barely at all the next. Thin apples to one fruit per cluster, about six inches apart, and peaches to about eight inches apart.

Pollination requirements vary. Most apples, pears and sweet cherries need a second compatible variety that blooms at the same time planted nearby. Peaches, nectarines, sour cherries and many plums are self-fertile and will fruit on their own.

Honeybees and native bees do most of the pollination work, so avoid spraying insecticides while trees are in bloom.

Frost during bloom is the most common cause of a failed crop. Plant trees on slopes where cold air drains away rather than in low frost pockets.
"""),
    Document("herbs-01-growing", "Growing Culinary Herbs", "herbs", """
Most Mediterranean herbs, including rosemary, thyme, oregano and sage, evolved in poor, dry, rocky soils. They need full sun and excellent drainage and taste best when not overfed or overwatered. They do very well in containers and raised beds.

Leafy annual herbs such as basil, cilantro, dill and parsley prefer richer, consistently moist soil. Basil loves heat and should not be planted out until nights are reliably warm. Cilantro bolts quickly in hot weather, so sow it in spring and autumn, and make successive sowings every three weeks.

Mint spreads aggressively by underground runners. Grow it in a pot, or in a bottomless container sunk into the ground, to keep it from taking over a bed.

Chives, thyme, oregano, sage and mint are perennial in most temperate climates and return each year. Rosemary is hardy only to about 20°F, so in colder areas grow it in a pot and bring it indoors for winter.
"""),
    Document("herbs-02-harvest", "Harvesting and Preserving Herbs", "herbs", """
Harvest herbs in the morning, after the dew dries but before the heat of the day, when their essential oil content is highest. Regular harvesting encourages bushy growth.

Pinch basil above a pair of leaves to make it branch, and remove flower buds to keep it producing leaves. Cut back perennial herbs such as thyme and oregano by up to a third, but don't cut into old woody stems that have no leaves, as they may not regrow.

To dry herbs, tie small bundles and hang them upside down in a warm, dark, well-ventilated place for one to two weeks, or use a dehydrator on its lowest setting. Store dried herbs in airtight jars away from light.

Basil, cilantro, parsley and chives keep more flavor frozen than dried. Chop them and freeze in ice cube trays covered with water or olive oil.
"""),
    Document("plan-01-rotation", "Crop Rotation for Vegetable Gardens", "garden_planning", """
Crop rotation means not growing plants from the same family in the same place year after year. It reduces the buildup of soilborne diseases and pests that specialize in one family, and it balances nutrient demands.

The main families to track are the nightshades (tomatoes, peppers, potatoes, eggplant), brassicas (cabbage, broccoli, kale, radishes), legumes (peas and beans), cucurbits (squash, cucumbers, melons), alliums (onions, garlic, leeks), and umbellifers (carrots, parsnips, celery).

A simple four-year rotation divides the garden into four sections and moves each group one section each year: legumes, followed by nitrogen-hungry brassicas, followed by nightshades and cucurbits, followed by root crops and alliums.

In a small garden, perfect rotation may be impossible. Even moving crops a few feet helps, and at minimum avoid planting nightshades in the same bed two years in a row.
"""),
    Document("plan-02-succession", "Succession Planting and Garden Calendars", "garden_planning", """
Succession planting keeps a garden productive all season. Instead of sowing a whole row of lettuce at once, sow a short row every two to three weeks, so that a new crop is maturing as the last one finishes.

Another form of succession is following one crop with another in the same space. Early spring crops such as peas, spinach and radishes can be followed by summer crops such as beans or squash, and then by autumn crops such as kale, turnips and garlic.

Build a planting calendar by counting backward from your average first frost date using each crop's days to maturity, and add a couple of weeks for slower growth as days shorten in autumn.

Record what you planted where and when in a garden journal. Notes on varieties, planting dates, yields and problems are invaluable for planning future seasons.
"""),
    Document("cover-01-crops", "Cover Crops and Green Manure", "cover_crops", """
Cover crops are planted to protect and improve soil rather than for harvest. They prevent erosion over winter, suppress weeds, add organic matter, and loosen compacted soil with their roots.

Legumes such as crimson clover, hairy vetch and field peas host bacteria in root nodules that fix nitrogen from the air, which becomes available to the next crop when the cover crop is turned in. Grasses and cereals such as winter rye and oats produce large amounts of biomass and scavenge leftover nitrogen so that it doesn't leach away.

Sow cover crops in late summer or early autumn after harvest, about four to six weeks before the first frost. Oats die over winter in cold climates, leaving an easy-to-plant mulch. Winter rye survives and grows vigorously in spring.

In spring, cut or mow the cover crop before it sets seed, and either dig it in or leave it on the surface as mulch. Wait two to three weeks before planting into turned-in green manure to let it begin decomposing.
"""),
    Document("season-01-extension", "Extending the Growing Season", "season_extension", """
Season extension techniques let you plant earlier in spring and harvest later into autumn. Floating row cover, a lightweight fabric laid directly over plants, provides two to four degrees of frost protection and keeps insects out.

Low tunnels are hoops of wire or PVC covered with row cover or clear plastic. They trap more heat than row cover alone but must be vented on sunny days to prevent overheating.

Cold frames are bottomless boxes with a clear lid, traditionally an old window. They are ideal for hardening off seedlings in spring and for growing lettuce, spinach and other cold-hardy greens well into winter. Open the lid on sunny days above about 45°F.

Hoop houses and unheated greenhouses provide the most protection, allowing year-round harvest of hardy greens in many climates. Dark containers of water inside act as thermal mass, absorbing heat during the day and releasing it at night.
"""),
    Document("tools-01-care", "Choosing and Caring for Garden Tools", "tools", """
A small set of good tools covers most garden work: a round-point shovel, a spading fork, a hard rake, a hoe, a hand trowel, bypass pruners and a watering can or hose with a breaker nozzle.

Bypass pruners, which cut with a scissor action, make clean cuts on living stems. Anvil pruners crush the stem and are better suited to dead wood.

Clean soil off tools after use and dry them to prevent rust. A bucket of sand mixed with a little oil makes an easy station for cleaning and conditioning shovel and hoe blades. Sharpen hoes, shovels and pruners with a file or whetstone at the start of each season; a sharp hoe slices weeds with far less effort.

Disinfect pruners with rubbing alcohol between cuts when removing diseased plant material to avoid spreading pathogens. Rub linseed oil into wooden handles each winter to keep them from drying out and splintering.
"""),
]

ALL_DOCS: List[Document] = COMPOSTING_DOCS + TOMATO_DOCS + BACKGROUND_DOCS


def _contains(text: str, term: str) -> bool:
    return re.search(re.escape(term), text, flags=re.IGNORECASE) is not None


def verify_no_leakage(docs: List[Document]) -> Dict[str, List[str]]:
    """Check that thin/absent topics appear only where the ground truth says.

    Returns topic -> list of doc_ids mentioning it. Raises ValueError on leakage.
    """
    thin_hosts: Dict[str, List[str]] = {}
    problems = []
    for topic, terms in LEAKAGE_TERMS.items():
        passage = THIN_PASSAGES.get(topic)
        hosts = []
        for doc in docs:
            text = doc.to_markdown()
            if passage:
                if passage in text:
                    hosts.append(doc.doc_id)
                    text = text.replace(passage, "")
            hits = [t for t in terms if _contains(text, t)]
            if hits:
                problems.append(f"{doc.doc_id} mentions {topic} terms outside the designated passage: {hits}")
        if passage and len(hosts) != 1:
            problems.append(f"thin topic {topic} should be hosted by exactly 1 document, found {hosts}")
        thin_hosts[topic] = hosts
    if problems:
        raise ValueError("Ground-truth leakage detected:\n  " + "\n  ".join(problems))
    return thin_hosts


def build_ground_truth(docs: List[Document]) -> dict:
    thin_hosts = verify_no_leakage(docs)
    topics = {}
    for name, meta in EVALUATED_TOPICS.items():
        if meta["tier"] == "full":
            doc_ids = [d.doc_id for d in docs if d.topic == name]
        elif meta["tier"] == "thin":
            doc_ids = thin_hosts[name]
        else:
            doc_ids = []
        entry = {"tier": meta["tier"], "description": meta["description"], "documents": doc_ids}
        if meta["tier"] == "thin":
            entry["passage"] = THIN_PASSAGES[name]
        topics[name] = entry

    background: Dict[str, List[str]] = {}
    for d in docs:
        if d.topic not in EVALUATED_TOPICS:
            background.setdefault(d.topic, []).append(d.doc_id)

    return {
        "domain": "home food gardening",
        "tiers": {
            "full": "multiple detailed documents dedicated to the topic",
            "thin": "a single 2-3 sentence passage buried in a document about something else",
            "absent": "zero documents; topic is outside the KB domain",
        },
        "topics": topics,
        "background_topics": background,
        "document_topics": {d.doc_id: d.topic for d in docs},
    }


def build(out_dir: Path = DEFAULT_OUT) -> dict:
    out_dir = Path(out_dir)
    ground_truth = build_ground_truth(ALL_DOCS)

    docs_dir = out_dir / "docs"
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True)
    for doc in ALL_DOCS:
        (docs_dir / f"{doc.doc_id}.md").write_text(doc.to_markdown(), encoding="utf-8")
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")
    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    gt = build(args.out)
    print(f"Wrote {len(ALL_DOCS)} documents to {args.out / 'docs'}")
    for name, t in gt["topics"].items():
        print(f"  {name:<26} {t['tier']:<7} {len(t['documents'])} doc(s)")
    print(f"  + {len(gt['background_topics'])} background topics, "
          f"{sum(len(v) for v in gt['background_topics'].values())} docs")


if __name__ == "__main__":
    main()
