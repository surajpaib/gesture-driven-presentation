
import pandas as pd
import xmltodict
import numpy as np

"""
XML file 1. layer:
keys: 'data'
XML file 2. layer:
keys: 'Label', 'FPS', 'Frame'
(Label is string, FPS is number, Frame has 3. layer)
XML file 3. layer:
array of lentgh: 36 (Depends on time)
XML file 4. layer:
keys: 'Avg_x', 'Avg_y', 'Avg_dist', 'Keypoint'
(Avg_x, Avg_y, Avg_dist are numbers, Keypoint has 5.layer)
XML file 5. layer:
array of lentgh: 18 (ID, X, Y, Confidence) (18: Count of keypoints
"""



'''
Convert data into TrainX (nr_of_example, nr_timestep, 12). 
@Onur: I misunderstood the nr_timestep. Actually it is the number of coordinates (or nr of sampled frame or each video ) the training data.
'''
def read_data(dic_filename):
    with open(dic_filename) as fd:
        doc = xmltodict.parse(fd.read())
    #############DataX##############
    dataX = []
    nr_timestep = 25 #Video is 30fps and at least 1 second. #TODO: A fixed value is not a wise choice.
    for idx in np.linspace(3, len(doc['data']['Frame'])-3, nr_timestep, dtype = int): # Remove the noise in the beginning and ending
        data_X = []
        for j in [2, 3, 4, 5, 6, 7]:  # The order must be consistent: left shoulder, left elbow, left wrist, right shoulder, right elbow, right wrist
            data_X.append(float(doc['data']['Frame'][idx]['Keypoint'][j - 1]['X']))
            data_X.append(float(doc['data']['Frame'][idx]['Keypoint'][j - 1]['Y']))
        dataX.append(data_X)
    dataX = np.vstack(dataX)

    #############DataY##############
    dataY=[]
    if 'Iprev' in dic_filename:
        dataY = [1, 0, 0, 0]
    elif 'reset' in dic_filename:
        dataY = [0, 1, 0, 0]
    elif 'rnext' in dic_filename:
        dataY = [0, 0, 1, 0]
    elif 'startstop' in dic_filename:
        dataY = [0, 0, 0, 1]
    else:
        dataY = [0, 0, 0, 0]

    return dataX, dataY

dic_filename = 'rnext.xml'
dataX, dataY = read_data(dic_filename)

#TODO: The desired trainX shape is (nr_example, nr_timestep (set as 25), 12(2D coords of 6keypoints)). The trainY shape is (nr_example, 4). Next step is to iterate all files and stack data.


#TODO: I checked the video length with the following, and it ranges in 25-90. However I met a bug in parsing xml file, See the following.
###################Iterate files and check length
# import os
# directory = '/Users/lizhaolin/Downloads/preprocessed_video_data/xml_files/LPrev/'
#
# lengths=[]
# for filename in os.listdir(directory):
#     if filename.endswith(".xml"):
#         print(filename)
#         fd = open(directory + filename)
#         doc = xmltodict.parse(fd.read())
#         print(len(doc['data']['Frame']))
#         lengths.append(len(doc['data']['Frame']))
#
# ######A bug in parsing this file
# fd = open(directory + 'lprev (1).xml')
# #doc = xmltodict.parse(fd.read()) #ERROR: , in parse  parser.Parse(xml_input, True) xml.parsers.expat.ExpatError: mismatched tag: line 1, column 243
