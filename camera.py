import numpy as np

# For the camera.
from picamera import PiCamera, PiVideoFrameType, Color, mmal

import ctypes as ct

# For image processing.
from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageEnhance

# For unique name generation.
from uuid import uuid1

# For timing.
from datetime import datetime, timedelta

from time import sleep, time

# For filesystem manipulation.
import os

import io

# When using os.rename we get this error when using /dev/shm:
# Invalid cross-device link: '/dev/shm/79089616-6e63-11ea-9b1e-00e04c3601f0.tmp' -> '/media/sdcard/37da26c8-6e63-11ea-a280-00e04c3601f0-1585118349.9037716.h264'
from shutil import move

from collections import deque

from statistics import mean

import logging

import threading


# https://github.com/raspberrypi/firmware/issues/1167#issuecomment-511798033
class PiCamera2(PiCamera):
    AWB_MODES = {
        'off':           mmal.MMAL_PARAM_AWBMODE_OFF,
        'auto':          mmal.MMAL_PARAM_AWBMODE_AUTO,
        'sunlight':      mmal.MMAL_PARAM_AWBMODE_SUNLIGHT,
        'cloudy':        mmal.MMAL_PARAM_AWBMODE_CLOUDY,
        'shade':         mmal.MMAL_PARAM_AWBMODE_SHADE,
        'tungsten':      mmal.MMAL_PARAM_AWBMODE_TUNGSTEN,
        'fluorescent':   mmal.MMAL_PARAM_AWBMODE_FLUORESCENT,
        'incandescent':  mmal.MMAL_PARAM_AWBMODE_INCANDESCENT,
        'flash':         mmal.MMAL_PARAM_AWBMODE_FLASH,
        'horizon':       mmal.MMAL_PARAM_AWBMODE_HORIZON,
        'greyworld':     ct.c_uint32(10)
        }


# Needed for PiTFT preview output
# https://picamera.readthedocs.io/en/release-1.10/faq.html#the-preview-doesn-t-work-on-my-pitft-screen
# https://www.raspberrypi.org/forums/viewtopic.php?t=89017
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')


zoomFactor = 5


# Initialization state (doesn't change after init is called):
outDir = None
outSubPath = '/unsynced'
tmpdir = None
target_frame_rate = None
# How often to switch files when recording, seconds
cliplen = None
bitrate = None
screenWidth = None
screenHeight = None

camera = None

pythonStartTime = time()
sessionUuid = None
fontPath = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'


# Internal changing state:
curRecordingFile = None
initializing = False
initialized = False
stopRequested = False
deinited = False

focusMode = False
awbMode = 'auto'

recordingStarted = False
recordingRunning = False
recordingThread = None
updateOverlaysThread = None

currentOverlay = None
currentOverlayLayer = 3

# State from javascript that changes all the time:
timecodeStr = None
batteryStr = None
syncCompleted = False
sdCardPresent = False
hwClockError = False
diskSpaceInfo = None
syncing = False
buttonHoldText = None
imminentAction = None
referenceImagePath = None
userActive = True
userTransitioningToInactive = False

referenceImageFailed = False


def blinkVisible():
    return True
#    return round(time() - pythonStartTime) % 2 == 0

def updateOverlays():
    global currentOverlay
    global currentOverlayLayer
    global referenceImageFailed

    logging.debug('updateOverlays')

    try:
        img = Image.new('RGBA', (screenWidth, screenHeight))

        def drawText(text, color, background, x = 0.5, y = 0.5):
            draw = ImageDraw.Draw(img)
            draw.font = ImageFont.truetype(fontPath, 80)
            w, h = draw.textsize(text, draw.font)
            x = (screenWidth-w) * x
            y = (screenHeight-h) * y
            draw.rectangle((x, y, x + w, y + h), fill=background)
            draw.text((x, y), text, color)

        if not userActive:
            drawText('PRESS ANY BUTTON', (255, 255, 255), '#de0000')
        elif focusMode:
            drawText('FOCUS MODE', (255, 255, 255), '#de000077', 0.5, 0.8)
        else:
            if referenceImagePath is not None and not referenceImageFailed:
                try:
                    referenceImgRaw = Image.open(referenceImagePath, 'r')
                    referenceImgWidth, referenceImgHeight = referenceImgRaw.size
                    ratio = screenWidth / referenceImgWidth
                    imgWidth = screenWidth
                    imgHeight = screenHeight * ratio
                    referenceImg = referenceImgRaw.resize((imgWidth, imgHeight))
                    referenceImg.putalpha(127)
                    img.paste(referenceImg)
                except Exception as e:
                    logging.exception("message")
                    referenceImageFailed = True

            circleSize = 150
            circlePosX = screenWidth - 20 - circleSize
            circlePosY = 20

            # Draw focus preview area indicator
            if not (recordingStarted or syncing or imminentAction):
                x = screenWidth / 2
                y = screenHeight / 2
                w = screenWidth / zoomFactor
                h = screenHeight / zoomFactor
                draw = ImageDraw.Draw(img, 'RGBA')   
                draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), outline='#de000077', width=int(screenWidth / 100))

            if batteryStr is not None:
                draw = ImageDraw.Draw(img)
                draw.font = ImageFont.truetype(fontPath, 60)
                w, h = draw.font.getsize(batteryStr)
                x = screenWidth - w - 10
                y = 10
                draw.rectangle((x, y, x + w, y + h), fill=('#865439'))
                draw.text((x, y), batteryStr, (255, 255, 255))

            if recordingStarted and blinkVisible():
                draw = ImageDraw.Draw(img)
                draw.ellipse((circlePosX, circlePosY, circlePosX + circleSize, circlePosY + circleSize), fill = '#de0000', outline ='#de0000')

            if syncing and blinkVisible():
                draw = ImageDraw.Draw(img)
                outline = 4
                draw.ellipse((circlePosX - outline, circlePosY - outline, circlePosX + circleSize + outline, circlePosY + circleSize + outline), fill = 'white')
                draw.ellipse((circlePosX, circlePosY, circlePosX + circleSize, circlePosY + circleSize), fill = '#0065ad')

            if timecodeStr is not None:
                draw = ImageDraw.Draw(img)
                draw.font = ImageFont.truetype(fontPath, 60)
                w, h = draw.font.getsize(timecodeStr)
                x = 10
                y = screenHeight - 60 - 10
                draw.rectangle((x, y, x + w, y + h), fill=('#0065ad' if syncCompleted else '#de0000'))
                draw.text((x, y), timecodeStr, (255, 255, 255))

            if diskSpaceInfo is not None:
                free = diskSpaceInfo['available']
                used = diskSpaceInfo['used']
                total = diskSpaceInfo['size']
                sizeStr = 'FREE ' + str(round(free / 1e9)) + ' / ' + str(round(total / 1e9)) + ' GB'
                draw = ImageDraw.Draw(img)
                draw.font = ImageFont.truetype(fontPath, 60)
                w, h = draw.font.getsize(sizeStr)
                x = screenWidth - w - 10
                y = screenHeight - 60 - 10
                draw.rectangle((x, y, x + w, y + h), fill=('#0065ad' if (used < 1e9) else '#de0000'))
                draw.text((x, y), sizeStr, (255, 255, 255))

            if imminentAction is not None:
                drawText(imminentAction, (255, 255, 255), '#de0000', 0.5, 0.2)
            elif buttonHoldText is not None:
                text = 'Release button to\n'
                text += buttonHoldText
                drawText(text, (255, 255, 255), '#000000', 0.5, 0.2)
            else:
                errorText = ''
                if hwClockError:
                    errorText += 'RTC ERROR\n'
                if not syncCompleted and not syncing:
                    errorText += 'NO SYNC\n'
                if not sdCardPresent:
                    errorText += 'NO STORAGE\n'

                errorText = errorText.rstrip()

                if len(errorText) > 0:
                    drawText(errorText, (255, 255, 255), '#de0000', 0.5, 0.2)

        # Swapping overlays
        # https://github.com/waveform80/picamera/issues/448
        currentOverlayLayer = (3 if currentOverlayLayer == 4 else 3)
        newoverlay = camera.add_overlay(img.tobytes(), layer=currentOverlayLayer)

        if currentOverlay is not None:
            camera.remove_overlay(currentOverlay)

        currentOverlay = newoverlay
    except Exception as e:
        logging.exception("message")

def updateOverlaysLoop():
    global userTransitioningToInactive

    while not deinited:
        sleep(0.5)
        # Disable overlay update when user inactive, due to possible memory leak each time updateOverlays is called
        if userActive:
            updateOverlays()
        
        # Allow one last render before transitioning to inactive
        if not userActive and userTransitioningToInactive:
            userTransitioningToInactive = False
            updateOverlays()

def getFullOutDir():
    return outDir + outSubPath

def ensureOutDir():
    fullOutDir = getFullOutDir()
    try:
        os.mkdir(fullOutDir)
    except FileExistsError:
        pass

def setOutSubDir(newOutSubDir):
    global outSubPath
    outSubPath = '/' + newOutSubDir

def init(od, td, fps, cl, br, w, h, sid, loglevel):
    global outDir
    global tmpdir
    global initializing
    global initialized
    global camera
    global target_frame_rate
    global cliplen
    global bitrate
    global screenWidth
    global screenHeight
    global currentOverlay
    global updateOverlaysThread
    global sessionUuid

    target_frame_rate = fps
    cliplen = cl
    outDir = od
    tmpdir = td
    bitrate = br
    screenWidth = w
    screenHeight = h
    sessionUuid = sid

    logging.basicConfig(level=loglevel)

    if initializing or initialized:
        raise Exception('Tried to init when already inited')

    initializing = True

    # Initialize an instance of the PiCamera class, which provides an interface to
    # the V2 camera module on the Raspberry Pi.
    camera = PiCamera2(resolution=(w, h), framerate=target_frame_rate, clock_mode='raw')

    # Enable camera preview on the primary display.
    camera.start_preview()

    # this MAY have fixed a bug where the camera crashes on startup
    sleep(0.5)
    #updateOverlays()

    initializing = False
    initialized = True

    updateOverlaysThread = threading.Thread(target=updateOverlaysLoop)
    updateOverlaysThread.start()

def deinit():
    global updateOverlaysThread
    global deinited

    if not initialized:
        raise Exception('Tried to deinit before initialized')

    deinited = True
    updateOverlaysThread.join()
    updateOverlaysThread = None

    camera.close()

def stopRecording():
    global stopRequested
    global recordingThread

    if not recordingStarted or stopRequested:
        return

    stopRequested = True
    recordingThread.join()
    recordingThread = None

    #updateOverlays()

def setFocusMode(b):
    global focusMode
    focusMode = b

    if camera:
        if b:
            zoomSize = 1 / zoomFactor
            zoomPos = (0.5 - zoomSize / 2)
            camera.zoom = (zoomPos, zoomPos, zoomSize, zoomSize)
        else:
            camera.zoom = (0, 0, 1, 1)


def toggleWhiteBalance():
    global awbMode
    # We need separate state because if not, it will crash when we try to read awb_mode (due to the PiCamera2 hack)
    if awbMode == 'auto':
        camera.awb_mode = 'greyworld'
        awbMode = 'greyworld'
    else:
        camera.awb_mode = 'auto'
        awbMode = 'auto'

def isRecording():
    return recordingStarted

def setTimecodeStr(str):
    global timecodeStr
    timecodeStr = str

def setBatteryStr(str):
    global batteryStr
    batteryStr = str

def setSyncCompleted(b):
    global syncCompleted
    syncCompleted = b

def setHwClockError(b):
    global hwClockError
    hwClockError = b

def setSdCardPresent(b):
    global sdCardPresent
    sdCardPresent = b

def setDiskSpaceInfo(i):
    global diskSpaceInfo
    diskSpaceInfo = i

def setSyncing(s):
    global syncing
    syncing = s

def setButtonHoldText(s):
    global buttonHoldText
    buttonHoldText = s

def setImminentAction(str):
    global imminentAction
    imminentAction = str

def setReferenceImagePath(str):
    global referenceImagePath
    global referenceImageFailed
    referenceImagePath = str
    referenceImageFailed = False

def setUserActive(b):
    global userActive
    global userTransitioningToInactive
    if userActive:
        userTransitioningToInactive = True
    userActive = b

def recordLoop():
    global stopRequested
    global recordingStarted
    global recordingRunning
    global curRecordingFile

    def getfilenameprefix():
        return getFullOutDir() + '/' + sessionUuid

    # Function to create a file with a unique name for saving H.264 video.
    def createTmpVidFile():
        outpath = tmpdir + '/' + str(uuid1()) + ".tmp"
        file = open(outpath, 'wb')
        return outpath, file

    def getPtsStr(thePts):
        if thePts is None:
            return "null"
        else:
            return "{:.5f}".format(thePts)

    def savetime(startPtsStr, stopPtsStr):
        outpath = getfilenameprefix() + '-' + startPtsStr + ".json"
        file = open(outpath, 'w')
        file.write('{\n')
        file.write('\t"video":\n\t{\n')
        file.write('\t\t"start": ' + startPtsStr + ',\n')
        file.write('\t\t"stop": ' + stopPtsStr + '\n')
        file.write('\t}\n')
        file.write('}\n')
        file.close()

    def finishRecording(startpts, stoppts):
        global curRecordingFile
        try:
            startPtsStr = getPtsStr(startpts)
            stopPtsStr = getPtsStr(stoppts)
            if curRecordingFile:
                curRecordingFile.close()
                curRecordingFile = None
                # raise Exception("This is a test")
                move(curRecordingPath, getfilenameprefix() + "-" + startPtsStr + "-" + stopPtsStr + ".h264")
                savetime(startPtsStr, stopPtsStr)
        except Exception:
            # Cleanup tmp file!
            if os.path.isfile(curRecordingPath):
                os.remove(curRecordingPath)
            raise

    def getframepts(frame):
        # https://www.raspberrypi.org/forums/viewtopic.php?t=169081#p1086515
        frametime = frame.timestamp
        cameratime = camera.timestamp
        if frametime is None or cameratime is None:
            return None

        systime = time()

        #return frametime / 1e6
        return ((frametime - cameratime) / 1e6) + systime

    def stopInternal():
        global stopRequested
        global recordingStarted
        global recordingRunning
        stopRequested = False
        recordingStarted = False
        if recordingRunning:
            recordingRunning = False
            pts = getframepts(camera.frame)
            camera.stop_recording()
            finishRecording(clip_start_pts, pts)


    try:
        clip_start_time = None
        clip_start_frame = None

        # Create a file to record into.
        curRecordingPath, curRecordingFile = createTmpVidFile()

        # Start recording using the H.264 format.
        camera.start_recording(curRecordingFile, format='h264', bitrate=bitrate)

        recordingRunning = True

        # Wait for actual frame to arrive
        while True:
            frame = camera.frame
            if frame.complete and frame.timestamp is not None:
                break
            logging.debug('skipping first frame')
            camera.wait_recording(0.001)

        #print('frame.index', frame.index)
        #print('camera.timestamp', camera.timestamp / 1e6)
        #if frame.timestamp is not None:
        #    print('frame.timestamp', frame.timestamp / 1e6)
        #    print('frame.timestamp - camera.timestamp', (frame.timestamp - camera.timestamp) / 1e6)
        #print('frame.frame_type', frame.frame_type)

        clip_start_frame = frame.index
        clip_start_time = time()
        clip_start_pts = getframepts(frame)

        cur_frame_time = None
        last_frame_time = None
        cur_frame_pts = None
        last_frame_pts = None
        last_frame_index = 0
        cur_frame_index = 0

        max_frame_intervals = target_frame_rate * 3
        prev_frame_intervals = deque([], max_frame_intervals)

        while True:
            # Check if there is an error, and wait for a time less than frame time. Note that this is not synchronized to frames coming in
            # so this polling is a bit of a hack and we will skip frames if they are coming on too fast
            camera.wait_recording(0.001)

            now = time()

            frame = camera.frame

            # Skip if frame hasn't changed
            if cur_frame_index == frame.index:
                continue

            frames_advanced = frame.index - cur_frame_index
            if frames_advanced > 1:
                logging.debug('Skipped ' + str(frames_advanced - 1) + ' frame(s)!')

            last_frame_index = cur_frame_index
            cur_frame_index = frame.index

            if not frame.complete:
                logging.debug('Frame not complete!')

            #if frame.timestamp is None:
            #    logging.debug('Timestamp was none')

            last_frame_time = cur_frame_time
            cur_frame_time = time()

            last_frame_pts = cur_frame_pts
            cur_frame_pts = getframepts(frame)

            # stop recording and return to the idle state.
            if stopRequested:
                logging.debug('Stop button pressed')
                stopInternal()
                break

            clip_rel_frame = cur_frame_index - clip_start_frame
            cliptime = now - clip_start_time

            target_frame_interval = 1 / target_frame_rate

            avg_frame_interval = None
            if last_frame_time is not None:
                frame_interval = cur_frame_time - last_frame_time
                prev_frame_intervals.append(frame_interval)
                avg_frame_interval = mean(list(prev_frame_intervals))

                pts_interval = None
                if last_frame_pts is not None and cur_frame_pts is not None:
                    pts_interval = cur_frame_pts - last_frame_pts

                logging.debug('Frame ' + str(frame.index) + ' ' + str(clip_rel_frame) + ' ' + 'pts-interval: ' + ('N/A' if pts_interval is None else str(round(pts_interval * 1000, 2))) + ' ' + 'interval: ' + str(round(frame_interval * 1000, 2)) + ' ' + 'avginterval: ' + str(round(avg_frame_interval * 1000, 2)) + ' ' + ('SPS' if frame.frame_type == PiVideoFrameType.sps_header else '   ') + ' ' + ('I' if frame.frame_type == PiVideoFrameType.key_frame else '   ') + ('P' if frame.frame_type == PiVideoFrameType.frame else '   '))

            max_frame_interval_error = target_frame_interval * 0.1
            is_framerate_stable = avg_frame_interval is not None and abs(avg_frame_interval - target_frame_interval) < max_frame_interval_error

            if not is_framerate_stable:
                logging.debug('Frame rate is unstable')

            # If the recording duration has reached the clip length, transfer
            # recording to a new file
            # Typically it is best to split an H.264 stream so that it starts with an SPS/PPS header.
            # if (frame.frame_type == PiVideoFrameType.sps_header):
            #     stop = now
            # Actually it doesn't seem to matter
            if cliptime >= cliplen: # and is_framerate_stable:
                logging.info('Starting file split at frame count ' + str(cur_frame_index) + '(' + str(clip_rel_frame) + ')')

                # Create a new file to record into after split
                newPath, newFile = createTmpVidFile()

                # When this returns, a keyframe/sps has been produced and the file has been split
                camera.split_recording(newFile)
                # because we are probably at a sps frame which has no timestamp, we have to wait for next timestamped frame 
                while True:
                    if camera.frame.timestamp is not None:
                        break
                    logging.debug('Waiting for frame that has timestamp...')
                    camera.wait_recording(0.0001) # Keep low to be sure we don't miss additional frames if there is a "glitch" where a lot of frames are coming in quickly

                # Now re-fetch these variables because frame has probably changed:
                frame_index = camera.frame.index
                frame = camera.frame

                pts = getframepts(frame)
                systime = time()

                logging.info('pts duration ' + str(systime - clip_start_time))
                logging.info('apparent duration ' + str(systime - clip_start_time))

                finishRecording(clip_start_pts, pts)

                clip_start_time = systime
                clip_start_frame = frame_index
                clip_start_pts = pts

                curRecordingFile = newFile
                curRecordingPath = newPath
    except Exception as e:
        logging.exception("Caught exception. Stopping recording")
        # If stopRequested, we have probably already tried stopping and this is what caused the exception
        if not stopRequested:
            stopInternal()

def startRecording():
    global recordingStarted
    global recordingThread
    global focusMode

    if not initialized:
        raise Exception('Tried to start rec before initialized')

    if recordingStarted:
        raise Exception('Tried to start rec while already recording')
    
    focusMode = False

    recordingStarted = True

    ensureOutDir()

    recordingThread = threading.Thread(target=recordLoop)
    recordingThread.start()

def captureFrame(captureFramePath):
    #if recordingStarted:
    #    raise Exception('Tried capture frame while recording')
    camera.capture(captureFramePath, format='jpeg', quality=30, use_video_port=True)

#    updateOverlays()

# testing code:
#init('/media/sdcard', '/media/sdcard', 30, 5 * 60, 5000000, 1280, 720, '1234', 'DEBUG')
#focusMode = True
#sleep(3)
#focusMode = False
#sleep(3)
#startRecording()
#sleep(60)
#stopRecording()
#sleep(3)
#deinit()
