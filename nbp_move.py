#  _  _            _               ___ _      _         ___                       _ _   _          
# | \| |_  _ _ __ | |__  ___ _ _  | _ \ |__ _| |_ ___  | _ \___ __ ___  __ _ _ _ (_) |_(_)___ _ _  
# | .` | || | '  \| '_ \/ -_) '_| |  _/ / _` |  _/ -_) |   / -_) _/ _ \/ _` | ' \| |  _| / _ \ ' \ 
# |_|\_|\_,_|_|_|_|_.__/\___|_|   |_| |_\__,_|\__\___| |_|_\___\__\___/\__, |_||_|_|\__|_\___/_||_|
#  

import random
from helperFunctions import *
import numpy as np

HARDCODED_PLATES_DIR = './hardcoded plates'

@setStaticVars(processed=None, scores=None, images=None, plateData=None)
def calculateMove(gamestate):
    ''' Calculate the move in the game '''
    plates = gamestate['Numberplates']
    choices = gamestate['Nchoices']

    # Deal with static variables
    if calculateMove.processed == None: # Initialise 'processed' memeory
        calculateMove.processed = [False,]*len(plates)
    if calculateMove.scores == None: # Initialise 'scores' memeory
        calculateMove.scores = [None,]*len(plates)
    if calculateMove.images == None: # Get the plate images
        print('Opening images')
        calculateMove.images = getPlateImages(plates)

    print('My: ' + str(gamestate['MyAnswers']))
    print('Op: ' + str(gamestate['OppAnswers']))
    # Change behaviour based on the state of the guesses
    if False and '' in gamestate['MyAnswers']: # Guess all the plates randomly
        notAnswered = list(i for i,x in enumerate(gamestate['MyAnswers']) if x == '')
        while True:
            plate = random.choice(notAnswered)
            choice = random.choice(choices)
            isOppAnswer = gamestate['OppAnswers'][plate] != choice
            isMyAnswer = choice in gamestate['MyAnswers']
            if not isOppAnswer and not isMyAnswer: break
        # IDEA: Guess all of these at once
        #       This would be possible only with multiple guesses per call
    elif False in calculateMove.processed: # Start analysing the plates

        # Deal with static variables
        if calculateMove.plateData == None: # Get the plate images
            # print('Creating plate data')
            # height of the image (60) * the ratio of the height of the letters to the plate (45/71)
            # plateData = list(getPlate2(guess,int(60*45/71)) for guess in choices)
            plateData = list(pil2np(getPlate(guess,int(60*45/71))) for guess in choices)
            calculateMove.plateData = plateData
        else:
            plateData = calculateMove.plateData

        # Choose a random unprocessed plate to process
        notProcessed = list(i for i,x in enumerate(calculateMove.processed) if x == False)
        print('nP: ' + str(notProcessed))
        plate = random.choice(notProcessed)
        image = calculateMove.images[plate]
         
        # Process the plate
        image = cropContourImg(image)
        c = getCorners(image)
        image = straightenImage(image, c)

        imageData = pil2np(image)
        # print('RMS Search')

        nToKeep = min((len(plates),len(choices)))
        scores, confidence = RMSMultiSearch(imageData, plateData, names=choices, skip=3, keep=nToKeep)

        choice = getBestChoice(gamestate, scores, plate)
        calculateMove.processed[plate] = True
    else: # Just update the guesses if the oponnent changes their mind
        plate = random.randint(0,len(plates)) # choose a random plate to update
        scores = calculateMove.scores[plate]
        choice = getBestChoice(gamestate, scores, plate)
    print({'Guess': choice,'Numberplate': plate})
    # return {'Guess': choice,'Numberplate': plate}
    return {'Guess': choice,'Numberplate': plate}

def getPlateImages(plates):
    '''
    This would pull the images from the links provided into a temp directory.
    For a time-being it uses the hard-coded images though
    '''
    return list(openGrayScale(im) for im in findAllInDir(HARDCODED_PLATES_DIR, sort=True))

def getBestChoice(gamestate, scores, plate):
    ''' Find the best choice given the gamestate and the scores
    '''
    # IDEA: Update the gamestate before this step
    # (in case the processig took a long time and the
    # move you're about to suggest is  actually invalid)
    for score, choice in scores:
        isOppAnswer = gamestate['OppAnswers'][plate] == choice
        isMyAnswer = choice in gamestate['MyAnswers']
        if isOppAnswer:
            pass
        elif isMyAnswer:
            # IDEA: Check the scores of two answers and pick the better one, also, update the other one
            #       This would be possible only with multiple guesses per call
            pass
        else:
            break
    return choice
