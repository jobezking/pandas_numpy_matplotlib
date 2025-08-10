import json
import pandas as pd

#open json file
with open('a_sample.json','r') as file:
    data = json.load(file)

print(data)

#Save to Json file

with open('output.json','w') as file:
    json.dump(data,file,indent=4)

#If json data is flat and tabular, pandas can read it directly

df = pd.read_json('output.json')
print(df.head())