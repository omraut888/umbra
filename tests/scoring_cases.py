"""20 hand-crafted (query, retrieved chunks) cases for scorer unit tests.

Expected labels use the spec §6 zone thresholds:
    good        -> Adequate zone, CS > 0.60   (chunks on-topic and one answers directly)
    bad         -> Dark zone,     CS < 0.30   (chunks from an unrelated domain)
    borderline  -> Thin zone,     0.30 <= CS <= 0.60
                   (on-topic chunks that don't contain the answer, or one relevant
                   chunk mixed in with unrelated ones)

The cases are deliberately independent of the synthetic KB and span several
domains, so the tests check the scorer rather than the KB.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ScoringCase:
    name: str
    label: str  # "good" | "bad" | "borderline"
    query: str
    chunks: List[str]


CASES: List[ScoringCase] = [
    ScoringCase("good_python_list_sort", "good",
        "How do I sort a list in Python in descending order?",
        [
            "To sort a list in descending order in Python, call list.sort(reverse=True) to sort in place, or use sorted(my_list, reverse=True) to return a new sorted list.",
            "Python's sorted() function accepts a key argument, such as key=len, to sort items by a computed value rather than the items themselves.",
            "The list.sort() method sorts a Python list in place and returns None, whereas sorted() works on any iterable and returns a new list.",
            "Python's sort is stable, meaning items that compare equal keep their original relative order, which makes multi-key sorting straightforward.",
        ]),
    ScoringCase("good_photosynthesis", "good",
        "What are the products of photosynthesis?",
        [
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen using energy from sunlight; glucose and oxygen are its products.",
            "The light-dependent reactions of photosynthesis take place in the thylakoid membranes and release oxygen as water molecules are split.",
            "In the Calvin cycle, plants fix carbon dioxide into three-carbon sugars that are later assembled into glucose.",
            "Chlorophyll absorbs mostly red and blue light, which powers the reactions of photosynthesis in the chloroplast.",
        ]),
    ScoringCase("good_http_404", "good",
        "What does an HTTP 404 status code mean?",
        [
            "HTTP 404 Not Found means the server could not find the requested resource; the URL may be wrong or the page may have been removed.",
            "HTTP status codes in the 4xx range indicate client errors, such as 400 Bad Request, 401 Unauthorized, 403 Forbidden and 404 Not Found.",
            "A 404 response differs from 410 Gone, which tells clients the resource was intentionally removed and will not return.",
            "Web servers can be configured to serve a custom 404 error page that helps users navigate back to working pages.",
        ]),
    ScoringCase("good_passport_renewal", "good",
        "How long does it take to renew a US passport?",
        [
            "Routine US passport renewal by mail currently takes about six to eight weeks, while expedited service takes two to three weeks for an additional fee.",
            "You can renew a US passport by mail with form DS-82 if your most recent passport was issued when you were 16 or older and within the last 15 years.",
            "Passport processing times do not include mailing time, so applicants should allow up to two extra weeks for delivery.",
            "Travelers with urgent international travel within 14 days can request an appointment at a passport agency for faster US passport service.",
        ]),
    ScoringCase("good_sourdough_starter", "good",
        "How often should I feed a sourdough starter?",
        [
            "A sourdough starter kept at room temperature should be fed once or twice a day, discarding some starter and adding equal weights of flour and water.",
            "A refrigerated sourdough starter only needs feeding about once a week; bring it back to room temperature and feed it before baking.",
            "A healthy sourdough starter doubles in size within four to eight hours of feeding and smells pleasantly sour.",
            "Liquid collecting on top of a sourdough starter, called hooch, is a sign that the starter is hungry and needs feeding.",
        ]),
    ScoringCase("good_git_undo_commit", "good",
        "How do I undo the last git commit but keep my changes?",
        [
            "Run git reset --soft HEAD~1 to undo the last commit while keeping its changes staged; use git reset HEAD~1 to keep them unstaged in your working tree.",
            "git reset --hard HEAD~1 undoes the last commit and discards its changes entirely, so use it with care.",
            "If the commit has already been pushed, git revert HEAD creates a new commit that undoes it without rewriting shared history.",
            "git commit --amend lets you change the most recent commit's message or add forgotten files to it.",
        ]),
    ScoringCase("good_vitamin_d", "good",
        "What are the main sources of vitamin D?",
        [
            "The main sources of vitamin D are sunlight on the skin, fatty fish such as salmon and mackerel, egg yolks, and fortified foods like milk and cereal.",
            "The skin produces vitamin D when exposed to ultraviolet B rays from the sun, which is the body's primary natural source of the vitamin.",
            "Vitamin D supplements come in two forms, D2 and D3, and D3 is generally more effective at raising blood levels.",
            "Vitamin D deficiency is more common in winter and at high latitudes, where sunlight is too weak for the skin to make enough.",
        ]),
    ScoringCase("bad_quantum_vs_baking", "bad",
        "What is quantum entanglement?",
        [
            "Cream the butter and sugar together until pale and fluffy before adding the eggs one at a time.",
            "Preheat the oven to 350 degrees Fahrenheit and grease a nine-inch round cake pan.",
            "Sift the flour with baking powder and a pinch of salt to avoid lumps in the batter.",
            "Let the cake cool in the pan for ten minutes before turning it out onto a wire rack.",
        ]),
    ScoringCase("bad_tax_vs_football", "bad",
        "How do I file an amended tax return?",
        [
            "The striker curled a free kick into the top corner in the ninetieth minute to win the match.",
            "The goalkeeper made three crucial saves in the penalty shootout.",
            "The team switched to a back three formation at halftime to push more players forward.",
            "Fans filled the stadium hours before kickoff for the derby.",
        ]),
    ScoringCase("bad_kubernetes_vs_poetry", "bad",
        "How do I scale a Kubernetes deployment?",
        [
            "Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
            "The sonnet's final couplet resolves the tension built up across the three quatrains.",
            "Romantic poets often turned to nature as a source of emotional and spiritual renewal.",
            "Iambic pentameter consists of five pairs of unstressed and stressed syllables per line.",
        ]),
    ScoringCase("bad_heart_attack_vs_cars", "bad",
        "What are the warning signs of a heart attack?",
        [
            "Check the tire pressure monthly and rotate the tires every five thousand miles.",
            "The alternator charges the battery while the engine is running.",
            "Change the engine oil and filter according to the manufacturer's schedule.",
            "Worn brake pads often produce a high-pitched squeal when braking.",
        ]),
    ScoringCase("bad_roman_empire_vs_skincare", "bad",
        "Why did the Western Roman Empire fall?",
        [
            "Apply sunscreen with an SPF of at least 30 every morning, even on cloudy days.",
            "Hyaluronic acid serums help the skin retain moisture.",
            "Exfoliate no more than two or three times a week to avoid irritation.",
            "Retinol can cause dryness at first, so introduce it gradually.",
        ]),
    ScoringCase("bad_mortgage_vs_birds", "bad",
        "How is a fixed-rate mortgage payment calculated?",
        [
            "The Arctic tern migrates from the Arctic to the Antarctic and back every year.",
            "Woodpeckers have reinforced skulls that absorb the shock of drilling into trees.",
            "Hummingbirds can hover and even fly backwards by rotating their wings in a figure eight.",
            "Owls hunt silently thanks to specialized feathers that muffle the sound of their wings.",
        ]),
    ScoringCase("bad_empty_retrieval", "bad",
        "What is the boiling point of ethanol?",
        []),
    ScoringCase("borderline_related_no_answer_compost_temp", "borderline",
        "What exact temperature must a compost pile reach to kill E. coli?",
        [
            "Compost piles should be turned regularly to supply oxygen to the microbes breaking down organic matter.",
            "A good compost mix contains roughly three parts brown material to one part green material by volume.",
            "Vegetable scraps, coffee grounds and grass clippings are common green materials for composting.",
            "Finished compost is dark and crumbly and smells earthy.",
        ]),
    ScoringCase("borderline_related_no_answer_python_gil", "borderline",
        "When was the Python GIL removed and in which release?",
        [
            "Python is a high-level programming language known for its readable syntax.",
            "Python's threading module lets programs run tasks concurrently.",
            "The multiprocessing module in Python starts separate processes that each have their own interpreter.",
            "Python's asyncio library provides an event loop for asynchronous programming.",
        ]),
    ScoringCase("borderline_one_relevant_among_noise_insulin", "borderline",
        "What does insulin do in the body?",
        [
            "Insulin is a hormone released by the pancreas that helps cells absorb glucose from the blood.",
            "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
            "Basketball was invented by James Naismith in 1891.",
            "Granite is an igneous rock composed mainly of quartz and feldspar.",
        ]),
    ScoringCase("borderline_one_relevant_among_noise_wifi", "borderline",
        "How do I change my Wi-Fi password on my router?",
        [
            "Log in to your router's admin page, usually at 192.168.1.1, to change wireless settings such as the network password.",
            "The mitochondria is the powerhouse of the cell.",
            "Mount Everest is the highest mountain above sea level.",
            "The French Revolution began in 1789.",
        ]),
    ScoringCase("borderline_adjacent_topic_electric_cars", "borderline",
        "How long does it take to charge a Tesla Model 3 at home?",
        [
            "Electric vehicles are becoming more popular as battery prices fall.",
            "Many countries offer tax incentives for buying an electric car.",
            "Electric motors deliver instant torque, giving EVs quick acceleration.",
            "Public charging networks are expanding along major highways.",
        ]),
    ScoringCase("borderline_adjacent_topic_marathon", "borderline",
        "What is the world record time for the women's marathon?",
        [
            "A marathon is a long-distance race of 42.195 kilometers.",
            "Marathon runners often follow training plans of sixteen to twenty weeks.",
            "Carbohydrate loading in the days before a marathon can improve performance.",
            "The Boston Marathon is one of the oldest annual marathons in the world.",
        ]),
]

assert len(CASES) == 20
