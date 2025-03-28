#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.5),
    on Fri 28 Mar 2025 05:27:11 PM CDT
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.5'
expName = 'mic_test'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [3440, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/group/jbinder/work/jheffernan/ForOthers/andanderson/mic_test/mic_test.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('record_key') is None:
        # initialise record_key
        record_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='record_key',
        )
    # initialise microphone
    deviceManager.addDevice(
        deviceClass='psychopy.hardware.microphone.MicrophoneDevice',
        deviceName='record_mic',
        index=None,
        maxRecordingSize=24000.0,
    )
    # create speaker 'playback_sound'
    deviceManager.addDevice(
        deviceName='playback_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    # Make folder to store recordings from record_mic
    record_micRecFolder = filename + '_record_mic_recorded'
    if not os.path.isdir(record_micRecFolder):
        os.mkdir(record_micRecFolder)
    
    # --- Initialize components for Routine "record" ---
    record_key = keyboard.Keyboard(deviceName='record_key')
    # make microphone object for record_mic
    record_mic = sound.microphone.Microphone(
        device='record_mic',
        name='record_mic',
        recordingFolder=record_micRecFolder,
        recordingExt='wav'
    )
    # Run 'Begin Experiment' code from record_code
    playback_gain = 1
    record_text = visual.TextStim(win=win, name='record_text',
        text='Press spacebar when finished recording.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "playback" ---
    playback_text = visual.TextStim(win=win, name='playback_text',
        text='Playing back sound.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    playback_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='playback_sound',    name='playback_sound'
    )
    playback_sound.setVolume(1.0)
    
    # --- Initialize components for Routine "adjust_sound" ---
    as_textbox = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.1, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='as_textbox',
         depth=0, autoLog=True,
    )
    as_text = visual.TextStim(win=win, name='as_text',
        text='Change sound level.',
        font='Arial',
        pos=(0, 0.0933), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    as_button = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0, -0.142),
        letterHeight=0.05,
        size=(0.4, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor='black',
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='as_button',
        depth=-2
    )
    as_button.buttonClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    sound_testing = data.TrialHandler2(
        name='sound_testing',
        nReps=10.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(sound_testing)  # add the loop to the experiment
    thisSound_testing = sound_testing.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSound_testing.rgb)
    if thisSound_testing != None:
        for paramName in thisSound_testing:
            globals()[paramName] = thisSound_testing[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSound_testing in sound_testing:
        currentLoop = sound_testing
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSound_testing.rgb)
        if thisSound_testing != None:
            for paramName in thisSound_testing:
                globals()[paramName] = thisSound_testing[paramName]
        
        # --- Prepare to start Routine "record" ---
        # create an object to store info about Routine record
        record = data.Routine(
            name='record',
            components=[record_key, record_mic, record_text],
        )
        record.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for record_key
        record_key.keys = []
        record_key.rt = []
        _record_key_allKeys = []
        # store start times for record
        record.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        record.tStart = globalClock.getTime(format='float')
        record.status = STARTED
        thisExp.addData('record.started', record.tStart)
        record.maxDuration = None
        # keep track of which components have finished
        recordComponents = record.components
        for thisComponent in record.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "record" ---
        # if trial has changed, end Routine now
        if isinstance(sound_testing, data.TrialHandler2) and thisSound_testing.thisN != sound_testing.thisTrial.thisN:
            continueRoutine = False
        record.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *record_key* updates
            waitOnFlip = False
            
            # if record_key is starting this frame...
            if record_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                record_key.frameNStart = frameN  # exact frame index
                record_key.tStart = t  # local t and not account for scr refresh
                record_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(record_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'record_key.started')
                # update status
                record_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(record_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(record_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if record_key.status == STARTED and not waitOnFlip:
                theseKeys = record_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _record_key_allKeys.extend(theseKeys)
                if len(_record_key_allKeys):
                    record_key.keys = _record_key_allKeys[-1].name  # just the last key pressed
                    record_key.rt = _record_key_allKeys[-1].rt
                    record_key.duration = _record_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # if record_mic is starting this frame...
            if record_mic.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                record_mic.frameNStart = frameN  # exact frame index
                record_mic.tStart = t  # local t and not account for scr refresh
                record_mic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(record_mic, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('record_mic.started', t)
                # update status
                record_mic.status = STARTED
                # start recording with record_mic
                record_mic.start()
            
            # if record_mic is active this frame...
            if record_mic.status == STARTED:
                # update params
                pass
                # update recorded clip for record_mic
                record_mic.poll()
            
            # if record_mic is stopping this frame...
            if record_mic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > record_mic.tStartRefresh + 15.625-frameTolerance:
                    # keep track of stop time/frame for later
                    record_mic.tStop = t  # not accounting for scr refresh
                    record_mic.tStopRefresh = tThisFlipGlobal  # on global time
                    record_mic.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('record_mic.stopped', t)
                    # update status
                    record_mic.status = FINISHED
                    # stop recording with record_mic
                    record_mic.stop()
            
            # *record_text* updates
            
            # if record_text is starting this frame...
            if record_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                record_text.frameNStart = frameN  # exact frame index
                record_text.tStart = t  # local t and not account for scr refresh
                record_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(record_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'record_text.started')
                # update status
                record_text.status = STARTED
                record_text.setAutoDraw(True)
            
            # if record_text is active this frame...
            if record_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                record.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in record.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "record" ---
        for thisComponent in record.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for record
        record.tStop = globalClock.getTime(format='float')
        record.tStopRefresh = tThisFlipGlobal
        thisExp.addData('record.stopped', record.tStop)
        # check responses
        if record_key.keys in ['', [], None]:  # No response was made
            record_key.keys = None
        sound_testing.addData('record_key.keys',record_key.keys)
        if record_key.keys != None:  # we had a response
            sound_testing.addData('record_key.rt', record_key.rt)
            sound_testing.addData('record_key.duration', record_key.duration)
        # tell mic to keep hold of current recording in record_mic.clips and transcript (if applicable) in record_mic.scripts
        # this will also update record_mic.lastClip and record_mic.lastScript
        record_mic.stop()
        tag = data.utils.getDateStr()
        record_micClip = record_mic.bank(
            tag=tag, transcribe='None',
            config=None
        )
        sound_testing.addData(
            'record_mic.clip', record_mic.recordingFolder / record_mic.getClipFilename(tag)
        )
        # Run 'End Routine' code from record_code
        record_micClip.gain(playback_gain)
        # the Routine "record" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "playback" ---
        # create an object to store info about Routine playback
        playback = data.Routine(
            name='playback',
            components=[playback_text, playback_sound],
        )
        playback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        playback_sound.setSound(record_micClip, secs=record_micClip.duration, hamming=True)
        playback_sound.setVolume(1.0, log=False)
        playback_sound.seek(0)
        # store start times for playback
        playback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        playback.tStart = globalClock.getTime(format='float')
        playback.status = STARTED
        thisExp.addData('playback.started', playback.tStart)
        playback.maxDuration = None
        # keep track of which components have finished
        playbackComponents = playback.components
        for thisComponent in playback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "playback" ---
        # if trial has changed, end Routine now
        if isinstance(sound_testing, data.TrialHandler2) and thisSound_testing.thisN != sound_testing.thisTrial.thisN:
            continueRoutine = False
        playback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *playback_text* updates
            
            # if playback_text is starting this frame...
            if playback_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                playback_text.frameNStart = frameN  # exact frame index
                playback_text.tStart = t  # local t and not account for scr refresh
                playback_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(playback_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'playback_text.started')
                # update status
                playback_text.status = STARTED
                playback_text.setAutoDraw(True)
            
            # if playback_text is active this frame...
            if playback_text.status == STARTED:
                # update params
                pass
            
            # if playback_text is stopping this frame...
            if playback_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > playback_text.tStartRefresh + record_micClip.duration-frameTolerance:
                    # keep track of stop time/frame for later
                    playback_text.tStop = t  # not accounting for scr refresh
                    playback_text.tStopRefresh = tThisFlipGlobal  # on global time
                    playback_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'playback_text.stopped')
                    # update status
                    playback_text.status = FINISHED
                    playback_text.setAutoDraw(False)
            
            # *playback_sound* updates
            
            # if playback_sound is starting this frame...
            if playback_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                playback_sound.frameNStart = frameN  # exact frame index
                playback_sound.tStart = t  # local t and not account for scr refresh
                playback_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('playback_sound.started', tThisFlipGlobal)
                # update status
                playback_sound.status = STARTED
                playback_sound.play(when=win)  # sync with win flip
            
            # if playback_sound is stopping this frame...
            if playback_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > playback_sound.tStartRefresh + record_micClip.duration-frameTolerance or playback_sound.isFinished:
                    # keep track of stop time/frame for later
                    playback_sound.tStop = t  # not accounting for scr refresh
                    playback_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    playback_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'playback_sound.stopped')
                    # update status
                    playback_sound.status = FINISHED
                    playback_sound.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[playback_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                playback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in playback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "playback" ---
        for thisComponent in playback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for playback
        playback.tStop = globalClock.getTime(format='float')
        playback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('playback.stopped', playback.tStop)
        playback_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "playback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "adjust_sound" ---
        # create an object to store info about Routine adjust_sound
        adjust_sound = data.Routine(
            name='adjust_sound',
            components=[as_textbox, as_text, as_button],
        )
        adjust_sound.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        as_textbox.reset()
        # reset as_button to account for continued clicks & clear times on/off
        as_button.reset()
        # store start times for adjust_sound
        adjust_sound.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        adjust_sound.tStart = globalClock.getTime(format='float')
        adjust_sound.status = STARTED
        thisExp.addData('adjust_sound.started', adjust_sound.tStart)
        adjust_sound.maxDuration = None
        # keep track of which components have finished
        adjust_soundComponents = adjust_sound.components
        for thisComponent in adjust_sound.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "adjust_sound" ---
        # if trial has changed, end Routine now
        if isinstance(sound_testing, data.TrialHandler2) and thisSound_testing.thisN != sound_testing.thisTrial.thisN:
            continueRoutine = False
        adjust_sound.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *as_textbox* updates
            
            # if as_textbox is starting this frame...
            if as_textbox.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                as_textbox.frameNStart = frameN  # exact frame index
                as_textbox.tStart = t  # local t and not account for scr refresh
                as_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(as_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'as_textbox.started')
                # update status
                as_textbox.status = STARTED
                as_textbox.setAutoDraw(True)
            
            # if as_textbox is active this frame...
            if as_textbox.status == STARTED:
                # update params
                pass
            
            # *as_text* updates
            
            # if as_text is starting this frame...
            if as_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                as_text.frameNStart = frameN  # exact frame index
                as_text.tStart = t  # local t and not account for scr refresh
                as_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(as_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'as_text.started')
                # update status
                as_text.status = STARTED
                as_text.setAutoDraw(True)
            
            # if as_text is active this frame...
            if as_text.status == STARTED:
                # update params
                pass
            # *as_button* updates
            
            # if as_button is starting this frame...
            if as_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                as_button.frameNStart = frameN  # exact frame index
                as_button.tStart = t  # local t and not account for scr refresh
                as_button.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(as_button, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'as_button.started')
                # update status
                as_button.status = STARTED
                win.callOnFlip(as_button.buttonClock.reset)
                as_button.setAutoDraw(True)
            
            # if as_button is active this frame...
            if as_button.status == STARTED:
                # update params
                pass
                # check whether as_button has been pressed
                if as_button.isClicked:
                    if not as_button.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        as_button.timesOn.append(as_button.buttonClock.getTime())
                        as_button.timesOff.append(as_button.buttonClock.getTime())
                    elif len(as_button.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        as_button.timesOff[-1] = as_button.buttonClock.getTime()
                    if not as_button.wasClicked:
                        # end routine when as_button is clicked
                        continueRoutine = False
                    if not as_button.wasClicked:
                        # run callback code when as_button is clicked
                        pass
            # take note of whether as_button was clicked, so that next frame we know if clicks are new
            as_button.wasClicked = as_button.isClicked and as_button.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                adjust_sound.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in adjust_sound.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "adjust_sound" ---
        for thisComponent in adjust_sound.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for adjust_sound
        adjust_sound.tStop = globalClock.getTime(format='float')
        adjust_sound.tStopRefresh = tThisFlipGlobal
        thisExp.addData('adjust_sound.stopped', adjust_sound.tStop)
        sound_testing.addData('as_textbox.text',as_textbox.text)
        sound_testing.addData('as_button.numClicks', as_button.numClicks)
        if as_button.numClicks:
           sound_testing.addData('as_button.timesOn', as_button.timesOn)
           sound_testing.addData('as_button.timesOff', as_button.timesOff)
        else:
           sound_testing.addData('as_button.timesOn', "")
           sound_testing.addData('as_button.timesOff', "")
        # Run 'End Routine' code from as_code
        # get textbox position
        # thisExp.addData('tb_pos0', as_text.getPos()[0])
        # thisExp.addData('tb_pos1', as_text.getPos()[1])
        
        try:
            playback_gain = float(as_textbox.text)
        except ValueError:
            playback_gain = 1
        thisExp.addData('playback_gain', playback_gain)
        
        # get polygon position
        # thisExp.addData('as_polygon0', as_polygon.getPos()[0])
        # thisExp.addData('as_polygon1', as_polygon.getPos()[1])
        # the Routine "adjust_sound" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 10.0 repeats of 'sound_testing'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # save record_mic recordings
    record_mic.saveClips()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
