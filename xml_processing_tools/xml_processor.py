
import pandas as pd
import xmltodict


"""
XML file 1. layer:
keys: 'data'

XML file 2. layer:
keys: 'Label', 'FPS', 'Frame'
(Label is string, FPS is number, Frame has 3. layer)

XML file 3. layer:
array of lentgh: 36 (Depends on time?)

XML file 4. layer:
keys: 'Avg_x', 'Avg_y', 'Avg_dist', 'Keypoint'
(Avg_x, Avg_y, Avg_dist are numbers, Keypoint has 5.layer)


XML file 5. layer:
array of lentgh: 18 (ID, X, Y, Confidence) (Depends on time?)
"""

with open('rnext.xml') as fd:
    doc = xmltodict.parse(fd.read())

print(doc)
print('############################################')
print(doc['data']['Frame'][0]['Avg_dist'])
print('############################################')
print(doc['data']['Frame'][0].keys())
print('############################################')
print(len(doc['data']['Frame']))
print('############################################')
print(doc['data'].keys())
