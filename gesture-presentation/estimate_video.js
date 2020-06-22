import * as posenet from '@tensorflow-models/posenet';
import * as handpose from '@tensorflow-models/handpose';

fs = require('fs')

async function detectOnFrame(video, pnet, hnet) {
    const pose = await pnet.estimatePoses(video, {
        flipHorizontal: flipPoseHorizontal,
        decodingMethod: 'single-person'
    });

    const hand = await hnet.estimatePoses(video);

    return [pose, hand]
}

function loadVideo(file) {
    var command = ffmpeg(file);
    return video
}

async function onLoad(num_vids) {
    const pnet = await posenet.load()
    const hnet = await handpose.load()
    var poses = []
    var hands = []
    for(var i; i < num_frames; i+=1){
        const detections = await detectOnFrame(loadVideo(), pnet, hnet)
        poses.append(detections[0])
        hands.append(detections[1])
    }
    fs.createWriteStream()

}