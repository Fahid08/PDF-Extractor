from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv('key.env')

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

delimiter = "####"
retrieval_message = """You will be given text from a pdf (containing unstructured text and structured tables) which will include ENTITIES AND THEIR CORRESPONDING VALUES, your objective is to extract those entities and their values from both text and tables and write them in a json dictionary as {entity:' ', value: ''}. Relevant text is given inside delimitter ####.
"""

text = """
Methane is produced microbially in anaerobic conditions and consumed in aerobic
conditions. In soils, the conditions can vary from anaerobic to aerobic in time or
space. The sites of CH4 production are the lower soil layers with low oxygen content
or soil aggregates favouring anaerobic bacteria. Sites of CH4 consumption are the
topsoil or macropores of the soil. Soil micro- or macroporosity was found to affect
the observed rates of CH4 flux in Finnish clay and sandy soils (Regina et al. 2007). In
cultivated soils, the annual balance of CH4 is usually close to zero most often
resulting in more CH4 being consumed than produced. Emissions of CH4 have
been reported to occur occasionally in wet conditions (Regina et al. 2007), but
even then, the annual balance typically indicates net consumption of CH4.
Compared to CO2, the carbon flows related to CH4 are minor (Table 2.2). The
annual fluxes have ranged from −0.12 to 0.06 g m−2 with no clear differences
between annual and perennial cropping can be seen. Grazing has been found to
change pastures from sink to source of CH4 emissions due to CH4 released from the
deposited dung (Maljanen et al. 2012b).
2.3.1.3 Nitrous Oxide
In cultivated mineral soils, the most significant gas in the total greenhouse gas
budget is N2O. Average annual emissions of N2O have been 0.6 g m−2 for annual
crops and 0.4 g m−2 for perennial crops including mostly grass leys
(Table 2.2). Annual emissions of N2O are typically slightly higher from annual
cultivation compared to perennial despite the higher fertilization rates on perennial
ley production (Regina et al. 2013).
Perennial crops take up nutrients clearly.
Table 2.2 Annual greenhouse gas fluxes of cultivated mineral soils
Mean (g m2 year−1) Min Max n Refs.
Annual crop
Net CO2 exchange –––
C loss as yield (CO2) –––
CH4 flux −0.04 ±0.07 −0.12 0.06 7 1; 2
N2O flux 0.57 ±0.23 0.20 1.02 31 1; 3; 4; 5
Perennial crop
Net CO2 exchangea −950 −961 −939 2 6
C loss as yield (CO2) 1183 1228 1137 2 6
CH4 flux −0.05 ±0.03 −0.09 0.03 14 1; 2; 10
N2O flux 0.43 ±0.31 0.06 1.13 20 1; 3; 4; 7; 8; 9
n = number of annual flux estimatesaNegative value = carbon sequestration, positive v
"""

model_name ="llama3-70b-8192" 
#model_name="mixtral-8x7b-32768"
chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{retrieval_message}{delimiter}{text}{delimiter}",
                    }
                    ],
                model=model_name,)

print(chat_completion.choices[0].message.content)
  
