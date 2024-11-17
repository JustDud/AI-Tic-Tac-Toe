# Importing all the libraries needed for the program to function
import os
import pygame
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Initialising pygame module
pygame.init()

# Declaring variables for the size of the window
width = 900
height = 640

# Defining the dimensions of the screen
screen = pygame.display.set_mode((width, height))

# Declaring the title of the window
pygame.display.set_caption('AI Tic Tac Toe')

# Creating variables that store colours used in the game
colour_white = (255, 255, 255)
colour_redDark = (220, 50, 90)
colour_redLight = (238, 66, 102)
colour_black = (0, 0, 0)

# This variable sets the screen that should be displayed initially
current_screen = "menu"

# Variable that stores human's and AI's characters
humanCharacter = 'x'
AICharacter = 'o'

# Keeps track whose move is next if turn = 0 - X, turn = 1 - O
turn = 0

# Variable that tracks if game mode has been picked
modePicked = False

# Variable that stores the number of moves made
moveCounter = 0

# variable to store the number of games played
gameCounter = 0

# This variable stores if gameCounter has been incremented
gameAdded = False

# This variable stores if auto mode has been picked
autoMode = False

# Loading images
current_dir = os.path.dirname(__file__)
images_dir = os.path.join(current_dir, '..', 'Images')

# Loading images
lineHorizontal = pygame.image.load(os.path.join(images_dir, "LineHorizontal.png")).convert()
lineVertical = pygame.image.load(os.path.join(images_dir, "LineVertical.png")).convert()
lineDiagonal1 = pygame.image.load(os.path.join(images_dir, "LineDiagonal(LtoR).png")).convert_alpha()
lineDiagonal2 = pygame.transform.rotate(lineDiagonal1, -90)

imgX = pygame.image.load(os.path.join(images_dir, "x.png")).convert()
imgO = pygame.image.load(os.path.join(images_dir, "o.png")).convert()

bg_menu = pygame.image.load(os.path.join(images_dir, "BG_Menu.png")).convert()
coatOfArms = pygame.image.load(os.path.join(images_dir, "coatOfArms.png")).convert()

bg_about = pygame.image.load(os.path.join(images_dir, "BG_About.png")).convert()
text_about = pygame.image.load(os.path.join(images_dir, "text_About.png")).convert()

settingsIcon = pygame.image.load(os.path.join(images_dir, "settingsIcon.png")).convert_alpha()
settingsBG = pygame.image.load(os.path.join(images_dir, "settingsBG.png")).convert()

# Variables to store the coordinates of the game counter number
gameXcoord = 0
gameYcoord = 0

# These variables are used to indicate whether the character has to be displayed or not
squareX1 = False
squareX2 = False
squareX3 = False
squareX4 = False
squareX5 = False
squareX6 = False
squareX7 = False
squareX8 = False
squareX9 = False

squareO1 = False
squareO2 = False
squareO3 = False
squareO4 = False
squareO5 = False
squareO6 = False
squareO7 = False
squareO8 = False
squareO9 = False

# These variables are used to block access to squares if there are characters in them
usedS1 = 0
usedS2 = 0
usedS3 = 0
usedS4 = 0
usedS5 = 0
usedS6 = 0
usedS7 = 0
usedS8 = 0
usedS9 = 0

# Victory combination booleans
victoryComb1 = False
victoryComb2 = False
victoryComb3 = False
victoryComb4 = False
victoryComb5 = False
victoryComb6 = False
victoryComb7 = False
victoryComb8 = False

# These variables are used to indicate which character is in which square
charSquare1 = ''
charSquare2 = ''
charSquare3 = ''
charSquare4 = ''
charSquare5 = ''
charSquare6 = ''
charSquare7 = ''
charSquare8 = ''
charSquare9 = ''

# Variables used to create a game grid
grid_width = 135
grid_height = 135
grid_spacing = 20
grid_firstPosX = 40
grid_firstPosY = 145

# Q table and related parameters declaration
# Number of actions and unique states (10 actions to make square number correspond to action taken)
num_actions = 10
num_states = 8533

# Initializing the Q-tables [state, action taken] with zeros
q_valuesAI = np.zeros((num_states, num_actions))
q_valuesHuman = np.zeros((num_states, num_actions))

# It determines to what extent newly acquired information overrides old information
learning_rate = 0.8

# Determines the trade-off between immediate and future rewards
discount_factor = 0.9

# The probability of choosing a random action over the action with the maximum Q-value
max_epsilon = 1
min_epsilon = 0.005
current_epsilonAI = 1
current_epsilonHuman = 1

# The rate at which epsilon will decrease
decay_rate = 0.0005

# Used to calculate epsilon value based on the current episode value and other parameters
current_episode = 1

# Variables that store rewards given to AI and Human (when in auto mode)
rewardAI = 0
rewardHuman = 0

# These are temporary variables that store q value and state id for previous state
old_q_valueAITemp = 0
oldStateIDAITemp = 0
action_indexAITemp = 0

old_q_valueHumanTemp = 0
oldStateIDHumanTemp = 0
action_indexHumanTemp = 0

# Initialising dictionary that stores states and their id's
statesDict = {}
stateID = 0

# Graph
# Sets the size of an image of the graph
plt.rcParams['figure.figsize'] = [4.3, 2.6]

# Lists that store values corresponding to x-axis
xAxis = [0]

# Lists that store values corresponding to y-axis
yAxisWins = [0]
yAxisLoses = [0]
yAxisDraw = [0]

# Used to update graph after a certain value of number of games has been reached
multiplier = 1

# A value that determines how often a new point on the graph is plotted
plotStep = 50

# These variables keep track of the number of games won, lost, and tied to plot the graph then
victoryNumAI = 0
defeatNumAI = 0
drawNum = 0

# Coordinates of graph's image location
graphXcoord = 480
graphYcoord = 350

# Used to shift clear board button and the game counter to show graph's image in a larger size
offset = -80

# Variables that store if the button in settings has been pressed
# Display Time
button1 = False
# Display Graph
button2 = False
# Output q values for current state
button3 = False
# Get q value for any state
button4 = False
# Set decay rate
button5 = False
# Set learning rate
button6 = False
# Set discount factor
button7 = False

# Allows to display graph if true
displayGraph = False

# Allows to output q values into terminal if true
outputQvalues = False

# Used to store start time to be able to determine time elapsed later
start_time = time.time()

# A boolean that tells if the time elapsed should be displayed
displayTime = False

# Coordinates at which time elapsed will be displayed
timeXcoord = 55
timeYcoord = 55

# Allows to display warning message below the "Clear Board" button if true
displayWarning = False

# Stores the (x,y) coordinates of the mouse
mouse = pygame.mouse.get_pos()

# Variable to keep the program running
running = True


def menu_screen():
    global current_screen
    global mouse

    # Filling background colour of the screen
    screen.fill(colour_white)

    # Displaying background image
    screen.blit(bg_menu, (0, 0))

    # Initialising fonts
    buttonFont = pygame.font.SysFont('Montserrat', 65, False, False)
    titleFont = pygame.font.SysFont('Montserrat', 85, False, False)
    subtitleFont = pygame.font.SysFont('Montserrat', 70, False, False)

    # Rendering text
    text_play = buttonFont.render('PLAY', True, colour_white)
    text_about = buttonFont.render('?', True, colour_white)
    text_title = titleFont.render('TIC TAC TOE', True, colour_white)
    text_subtitle = subtitleFont.render('AI VS YOU', True, colour_white)

    # if mouse is hovered on the Play button it changes its colour to a darker shade
    if 65 <= mouse[0] <= 365 and 503 <= mouse[1] <= 603:
        pygame.draw.rect(screen, colour_redDark, [65, 503, 300, 100], 0, 0, 25, 25, 25, 25)
    else:
        pygame.draw.rect(screen, colour_redLight, [65, 503, 300, 100], 0, 0, 25, 25, 25, 25)

    # if mouse is hovered on the About button it changes its colour to a darker shade
    if 810 <= mouse[0] <= 890 and 10 <= mouse[1] <= 90:
        pygame.draw.circle(screen, colour_redDark, [850, 50], 40)
    else:
        pygame.draw.circle(screen, colour_redLight, [850, 50], 40)

    # if mouse is hovered in the bottom right corner a hidden feature will be displayed
    if 860 <= mouse[0] <= 890 and 590 <= mouse[1] <= 637:
        screen.blit(coatOfArms, (860, 590))

    # locating text
    screen.blit(text_play, (160, 533))
    screen.blit(text_title, (45, 160))
    screen.blit(text_subtitle, (45, 225))
    screen.blit(text_about, (837, 30))

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            # if the mouse is clicked on the PLAY button, the game screen is opened
            if 65 <= mouse[0] <= 365 and 503 <= mouse[1] <= 603:
                current_screen = "game"
            # if the mouse is clicked on the ABOUT button, the about screen is opened
            if 810 <= mouse[0] <= 890 and 10 <= mouse[1] <= 90:
                current_screen = "about"
        # if the mouse is clicked on the window close button, the program is stopped
        if event.type == pygame.QUIT:
            current_screen = "quit"


def about_screen():
    global current_screen
    global mouse

    # Displaying background image
    screen.blit(bg_about, (0, 0))
    pygame.draw.rect(screen, colour_white, [125, 145, 650, 320], 0, 0, 25, 25, 25, 25)
    screen.blit(text_about, (135, 155))

    # font initialisation
    buttonFont = pygame.font.SysFont('Montserrat', 65, False, False)

    # rendering text
    text_close = buttonFont.render('X', True, colour_white)

    # if mouse is hovered on the About button it changes its colour to a darker shade
    if 735 <= mouse[0] <= 805 and 115 <= mouse[1] <= 185:
        pygame.draw.circle(screen, colour_redDark, [770, 150], 35)
    else:
        pygame.draw.circle(screen, colour_redLight, [770, 150], 35)

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            # if mouse is clicked on the About button, the menu screen is opened
            if 735 <= mouse[0] <= 805 and 115 <= mouse[1] <= 185:
                current_screen = "menu"
        # if mouse is clicked on the window close button, the program is stopped by setting current screen to quit
        if event.type == pygame.QUIT:
            current_screen = "quit"

    # locating text
    screen.blit(text_close, (756, 129))


def settings_screen():
    global current_screen
    global mouse
    global gameCounter
    global stateID

    global button1
    global button2
    global button3
    global button4
    global button5
    global button6
    global button7

    global displayTime
    global displayGraph
    global outputQvalues

    global decay_rate
    global learning_rate
    global decay_rate
    global discount_factor

    # Variables to set size of the buttons
    buttonWidth = 120
    buttonHeight = 45

    # Variables to set button locations
    buttonXcoord = 690
    buttonYcoord = 108
    spacing = buttonHeight + 15

    # Variable to control text spacing
    text_spacing = 60

    # Displaying background image
    screen.blit(settingsBG, (0, 0))

    # Font initialisation
    closeFont = pygame.font.SysFont('Montserrat', 65, False, False)
    buttonFont = pygame.font.SysFont('Montserrat', 40, False, False)

    # Rendering text
    text_close = closeFont.render('X', True, colour_white)
    text_buttonOn = buttonFont.render('ON', True, colour_white)
    text_buttonOff = buttonFont.render('OFF', True, colour_white)
    text_buttonGet = buttonFont.render('GET', True, colour_white)
    text_buttonSet = buttonFont.render('SET', True, colour_white)


    # if mouse is hovered on the button, it changes its colour to a darker shade
    # close button
    if 805 <= mouse[0] <= 875 and 25 <= mouse[1] <= 95:
        pygame.draw.circle(screen, colour_redDark, [840, 60], 35)
    else:
        pygame.draw.circle(screen, colour_redLight, [840, 60], 35)

    # button 1
    if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord <= mouse[1] <= buttonYcoord + buttonHeight:
        pygame.draw.rect(screen, colour_redDark, [buttonXcoord, buttonYcoord, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [buttonXcoord, buttonYcoord, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    # button 2
    if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + spacing <= mouse[1] <= buttonYcoord + buttonHeight + spacing:
        pygame.draw.rect(screen, colour_redDark, [buttonXcoord, buttonYcoord+spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [buttonXcoord, buttonYcoord+spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    # button 3
    if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 2*spacing <= mouse[1] <= buttonYcoord + buttonHeight + 2*spacing:
        pygame.draw.rect(screen, colour_redDark, [buttonXcoord, buttonYcoord+2*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [buttonXcoord, buttonYcoord+2*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    # button 4
    if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 3*spacing <= mouse[1] <= buttonYcoord + buttonHeight + 3*spacing:
        pygame.draw.rect(screen, colour_redDark, [buttonXcoord, buttonYcoord+3*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [buttonXcoord, buttonYcoord+3*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    # button 5
    if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 5*spacing <= mouse[1] <= buttonYcoord + buttonHeight + 5*spacing:
        pygame.draw.rect(screen, colour_redDark, [buttonXcoord, buttonYcoord+5*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [buttonXcoord, buttonYcoord+5*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    # button 6
    if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 6*spacing <= mouse[1] <= buttonYcoord + buttonHeight + 6*spacing:
        pygame.draw.rect(screen, colour_redDark, [buttonXcoord, buttonYcoord+6*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [buttonXcoord, buttonYcoord+6*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    # button 7
    if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 7*spacing <= mouse[1] <= buttonYcoord + buttonHeight + 7*spacing:
        pygame.draw.rect(screen, colour_redDark, [buttonXcoord, buttonYcoord+7*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [buttonXcoord, buttonYcoord+7*spacing, buttonWidth, buttonHeight], 0, 0, 12, 12, 12, 12)

    # locating text
    screen.blit(text_close, (825, 39))

    # Displaying text on the buttons
    if button1:
        screen.blit(text_buttonOn, (buttonXcoord + 40, buttonYcoord + 9))
    else:
        screen.blit(text_buttonOff, (buttonXcoord + 35, buttonYcoord + 11))
    if button2:
        screen.blit(text_buttonOn, (buttonXcoord + 40, buttonYcoord + 9 + text_spacing))
    else:
        screen.blit(text_buttonOff, (buttonXcoord + 35, buttonYcoord + 11 + text_spacing))
    if button3:
        screen.blit(text_buttonOn, (buttonXcoord + 40, buttonYcoord + 9 + 2 * text_spacing))
    else:
        screen.blit(text_buttonOff, (buttonXcoord + 35, buttonYcoord + 11 + 2 * text_spacing))

    screen.blit(text_buttonGet, (buttonXcoord + 35, buttonYcoord + 9 + 3 * text_spacing))
    screen.blit(text_buttonSet, (buttonXcoord + 35, buttonYcoord + 9 + 5 * text_spacing))
    screen.blit(text_buttonSet, (buttonXcoord + 35, buttonYcoord + 9 + 6 * text_spacing))
    screen.blit(text_buttonSet, (buttonXcoord + 35, buttonYcoord + 9 + 7 * text_spacing))

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            # button 1
            if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord <= mouse[1] <= buttonYcoord + buttonHeight:
                if button1:
                    button1 = False
                    displayTime = False
                else:
                    button1 = True
                    displayTime = True
            # button 2
            if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + spacing <= mouse[1] <= buttonYcoord + buttonHeight + spacing:
                if not button2 and gameCounter >= 50:
                    button2 = True
                    displayGraph = True
                else:
                    button2 = False
                    displayGraph = False
            # button 3
            if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 2 * spacing <= mouse[1] <= buttonYcoord + buttonHeight + 2 * spacing:
                if button3:
                    button3 = False
                    outputQvalues = False
                else:
                    button3 = True
                    outputQvalues = True
            # button 4
            if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 3 * spacing <= mouse[1] <= buttonYcoord + buttonHeight + 3 * spacing:
                if not button4:
                    button4 = True
                    stop = False
                    # loop to ask user for an input and validate it, till exit command is given
                    while not stop:
                        try:
                            print("Number of unique states:", stateID)
                            userStateID = int(input("Enter ID of the state you want to see the Q Table for or -1 to exit: "))
                            if userStateID == -1:
                                print("You may return back to settings")
                                break
                            elif userStateID >= 0:
                                if userStateID <= stateID:
                                    print(q_valuesAI[userStateID])
                                    print("")
                                else:
                                    print("State with entered id does not exist, try again or enter -1 to exit")
                                    print("")
                            else:
                                print("\nInvalid value, try again. Enter -1 to exit without making any changes\n")
                        except ValueError:
                            print("\nInvalid id, try again. Enter -1 to exit\n")
                    button4 = False

            if moveCounter == 0 and gameCounter == 0:
                # button 5
                if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 5 * spacing <= mouse[1] <= buttonYcoord + buttonHeight + 5 * spacing:
                    if not button5:
                        button5 = True
                        stop = False
                        # loop to ask for input and validate it, till exit command is given
                        while not stop:
                            try:
                                userDecayRate = float(input("Enter value for decay rate (range: from 0 to 1) or -1 to exit: "))
                                if userDecayRate == -1:
                                    print("You may return back to settings")
                                    break
                                if 0 <= userDecayRate <= 1:
                                    stop = True
                                    decay_rate = userDecayRate
                                    print("Changes have successfully been applied. You may return back to settings")
                                else:
                                    print("\nInvalid value, try again. Enter -1 to exit without making any changes\n")
                            except ValueError:
                                print("\nInvalid value, try again. Enter -1 to exit without making any changes\n")
                                print("")
                        button5 = False
                # button 6
                if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 6 * spacing <= mouse[1] <= buttonYcoord + buttonHeight + 6 * spacing:
                    if not button6:
                        button6 = True
                        stop = False
                        # loop to ask for input and validate it, till exit command is given
                        while not stop:
                            try:
                                userLearningRate = float(input("Enter value for learning rate (range: from 0 to 1) or -1 to exit: "))
                                if userLearningRate == -1:
                                    print("You may return back to settings")
                                    break
                                if 0 <= userLearningRate <= 1:
                                    stop = True
                                    learning_rate = userLearningRate
                                    print("Changes have successfully been applied. You may return back to settings")
                                else:
                                    print("\nInvalid value, try again. Enter -1 to exit without making any changes\n")
                            except ValueError:
                                print("\nInvalid value, try again. Enter -1 to exit without making any changes\n")
                        button6 = False
                # button 7
                if buttonXcoord <= mouse[0] <= buttonXcoord + buttonWidth and buttonYcoord + 7 * spacing <= mouse[1] <= buttonYcoord + buttonHeight + 7 * spacing:
                    if not button7:
                        button7 = True
                        stop = False
                        # loop to ask for input and validate it, till exit command is given
                        while not stop:
                            try:
                                userDiscountFactor = float(input("Enter value for learning rate (range: from 0 to 1) or -1 to exit: "))
                                if userDiscountFactor == -1:
                                    print("You may return back to settings")
                                    break
                                if 0 <= userDiscountFactor <= 1:
                                    stop = True
                                    discount_factor = userDiscountFactor
                                    print("Changes have successfully been applied. You may return back to settings")
                                else:
                                    print("\nInvalid value, try again. Enter -1 to exit without making any changes\n")
                            except ValueError:
                                print("\nInvalid value, try again. Enter -1 to exit without making any changes\n")
                        button7 = False

            # if the mouse is clicked on the close button, the game screen is reopened
            if 805 <= mouse[0] <= 875 and 25 <= mouse[1] <= 95:
                if not button4 or not button5 or not button6 or not button7:
                    current_screen = "game"

        if event.type == pygame.QUIT:
            # if mouse is clicked on the window close button, the program is stopped by setting current screen to quit
            if not button4 or not button5 or not button6 or not button7:
                current_screen = "quit"


def game_screen():
    global current_screen
    global mouse
    global gameCounter
    global moveCounter
    global modePicked
    global autoMode
    global turn

    global humanCharacter
    global AICharacter

    global gameXcoord
    global gameYcoord
    global timeXcoord
    global timeYcoord

    global displayWarning
    global displayTime
    global displayGraph
    global outputQvalues

    global plotStep
    global multiplier

    # Fill background colour of the screen
    screen.fill(colour_white)

    # Initialising font
    titleFont = pygame.font.SysFont('Montserrat', 35, False, False)
    textFont = pygame.font.SysFont('Montserrat', 35, False, False)
    text2Font = pygame.font.SysFont('Montserrat', 40, False, False)
    text3Font = pygame.font.SysFont('Montserrat', 25, False, False)

    # Rendering text
    text_title = titleFont.render('Choose training mode', True, colour_black)
    text_manual = text2Font.render('Manual', True, colour_white)
    text_auto = text2Font.render('Auto', True, colour_white)
    text_counter1 = textFont.render('Number of games', True, colour_black)
    text_counter2 = textFont.render('%s' % gameCounter, True, colour_black)
    text_counter3 = textFont.render('AI is trained on:', True, colour_black)
    text_clearButton = textFont.render('Clear Board', True, colour_white)
    text_warning = text3Font.render('Finish the game to clear the board', True, colour_black)
    text_time1 = textFont.render('Time elapsed:', True, colour_black)
    text_time2 = textFont.render(timeElapsed(), True, colour_black)

    # if mouse is hovered on the Manual button it changes its colour to a darker shade
    if 230 <= mouse[0] <= 420 and 50 <= mouse[1] <= 100:
        pygame.draw.rect(screen, colour_redDark, [230, 57, 190, 50], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [230, 57, 190, 50], 0, 0, 12, 12, 12, 12)

    # if mouse is hovered on the Auto button it changes its colour to a darker shade
    if 450 <= mouse[0] <= 640 and 50 <= mouse[1] <= 100:
        pygame.draw.rect(screen, colour_redDark, [450, 57, 190, 50], 0, 0, 12, 12, 12, 12)
    else:
        pygame.draw.rect(screen, colour_redLight, [450, 57, 190, 50], 0, 0, 12, 12, 12, 12)

    # if mouse is hovered on the Clear Board button it changes its colour to a darker shade
    if not displayGraph:
        if 605 <= mouse[0] <= 795 and 350 <= mouse[1] <= 400:
            pygame.draw.rect(screen, colour_redDark, [605, 350, 190, 50], 0, 0, 12, 12, 12, 12)
        else:
            pygame.draw.rect(screen, colour_redLight, [605, 350, 190, 50], 0, 0, 12, 12, 12, 12)
    else:
        if 605 <= mouse[0] <= 795 and 350 + offset <= mouse[1] <= 400 + offset:
            pygame.draw.rect(screen, colour_redDark, [605, 350 + offset, 190, 50], 0, 0, 12, 12, 12, 12)
        else:
            pygame.draw.rect(screen, colour_redLight, [605, 350 + offset, 190, 50], 0, 0, 12, 12, 12, 12)

    # if mouse is hovered on the settings button it changes its colour to a darker shade
    if 830 <= mouse[0] <= 890 and 10 <= mouse[1] <= 70:
        pygame.draw.circle(screen, colour_redDark, [860, 40], 30)
    else:
        pygame.draw.circle(screen, colour_redLight, [860, 40], 30)


    # Displaying settings icon
    screen.blit(settingsIcon, (839, 19))

    if modePicked:
        # Once the mode has been picked, the button of that mode will be displayed in a darker shade
        if not autoMode:
            pygame.draw.rect(screen, colour_redDark, [230, 57, 190, 50], 0, 0, 12, 12, 12, 12)
            pygame.draw.rect(screen, colour_redLight, [450, 57, 190, 50], 0, 0, 12, 12, 12, 12)

        elif autoMode:
            pygame.draw.rect(screen, colour_redDark, [450, 57, 190, 50], 0, 0, 12, 12, 12, 12)
            pygame.draw.rect(screen, colour_redLight, [230, 57, 190, 50], 0, 0, 12, 12, 12, 12)

    # Locating text
    gameTextLocation(gameCounter)
    screen.blit(text_title, (302, 17))
    screen.blit(text_manual, (277, 68))
    screen.blit(text_auto, (514, 68))

    # If button 2 (display graph) is pressed, clear board button, game counter, and some texts will be moved up by an offset
    if not displayGraph:
        screen.blit(text_counter1, (598, 250))
        screen.blit(text_counter2, (gameXcoord, gameYcoord))
        screen.blit(text_counter3, (610, 280))
        screen.blit(text_clearButton, (630, 363))
        if displayWarning:
            screen.blit(text_warning, (560, 405))
    else:
        screen.blit(text_counter1, (598, 250 + offset))
        screen.blit(text_counter2, (gameXcoord, gameYcoord + offset))
        screen.blit(text_counter3, (610, 280 + offset))
        screen.blit(text_clearButton, (630, 363 + offset))
        if displayWarning:
            screen.blit(text_warning, (560, 405 + offset))

    # Displaying time elapsed and text if display time button is pressed
    if displayTime:
        screen.blit(text_time1, (timeXcoord - 30, timeYcoord - 35))
        screen.blit(text_time2, (timeXcoord, timeYcoord))

    # Calling function which displays the grid
    grid()

    for event in pygame.event.get():
        # Checks if mouse is clicked
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If mouse is clicked on Clear Board button
            if not displayGraph:
                if 605 <= mouse[0] <= 795 and 350 <= mouse[1] <= 400:
                    if resultDetection() or moveCounter == 9:
                        clearBoard()
                    if moveCounter > 0:
                        displayWarning = True
            # When graph is displayed, clear board button is shifted and so its hit box
            else:
                if 605 <= mouse[0] <= 795 and 350 + offset <= mouse[1] <= 400 + offset:
                    if resultDetection() or moveCounter == 9:
                        clearBoard()
                    else:
                        displayWarning = True

            # If mouse is clicked on the MANUAL mode
            if 230 <= mouse[0] <= 420 and 50 <= mouse[1] <= 100:
                modePicked = True
                autoMode = False

            # If mouse is clicked on the AUTO mode
            elif 450 <= mouse[0] <= 640 and 50 <= mouse[1] <= 100:
                modePicked = True
                autoMode = True

            # If mouse is clicked on the SETTINGS button
            if 830 <= mouse[0] <= 890 and 10 <= mouse[1] <= 70:
                current_screen = 'settings'

            # If in manual mode, human will be asked to make a move
            if not autoMode and not resultDetection() and modePicked:
                if turn == 0:
                    if humanCharacter == 'x':
                        # if the mouse is clicked on the square, X character is displayed in it
                        # row 1
                        if grid_firstPosX <= mouse[0] <= grid_firstPosX + grid_width and grid_firstPosY <= mouse[1] <= grid_firstPosY + grid_height:
                            moveHuman('x', 1)

                        elif grid_firstPosX + grid_width + grid_spacing <= mouse[0] <= grid_firstPosX + 2*grid_width + grid_spacing and grid_firstPosY <= mouse[1] <= grid_firstPosY + grid_height:
                            moveHuman('x', 2)
                        elif grid_firstPosX + 2*grid_width + 2*grid_spacing <= mouse[0] <= grid_firstPosX + 3*grid_width + 2*grid_spacing and grid_firstPosY <= mouse[1] <= grid_firstPosY + grid_height:
                            moveHuman('x', 3)
                        # row 2
                        elif grid_firstPosX <= mouse[0] <= grid_firstPosX + grid_width and grid_firstPosY + grid_height + grid_spacing <= mouse[1] <= grid_firstPosY + 2*grid_height + grid_spacing:
                            moveHuman('x', 4)
                        elif grid_firstPosX + grid_width + grid_spacing <= mouse[0] <= grid_firstPosX + 2*grid_width + grid_spacing and grid_firstPosY + grid_height + grid_spacing <= mouse[1] <= grid_firstPosY + 2*grid_height + grid_spacing:
                            moveHuman('x', 5)
                        elif grid_firstPosX + 2*grid_width + 2*grid_spacing <= mouse[0] <= grid_firstPosX + 3*grid_width + 2*grid_spacing and grid_firstPosY + grid_height + grid_spacing <= mouse[1] <= grid_firstPosY + 2*grid_height + grid_spacing:
                            moveHuman('x', 6)
                        # row 3
                        elif grid_firstPosX <= mouse[0] <= grid_firstPosX + grid_width and grid_firstPosY + 2*grid_height + 2*grid_spacing <= mouse[1] <= grid_firstPosY + 3*grid_height + 2*grid_spacing:
                            moveHuman('x', 7)
                        elif grid_firstPosX + grid_width + grid_spacing <= mouse[0] <= grid_firstPosX + 2*grid_width + grid_spacing and grid_firstPosY + 2*grid_height + 2*grid_spacing <= mouse[1] <= grid_firstPosY + 3*grid_height + 2*grid_spacing:
                            moveHuman('x', 8)
                        elif grid_firstPosX + 2*grid_width + 2*grid_spacing <= mouse[0] <= grid_firstPosX + 3*grid_width + 2*grid_spacing and grid_firstPosY + 2*grid_height + 2*grid_spacing <= mouse[1] <= grid_firstPosY + 3*grid_height + 2*grid_spacing:
                            moveHuman('x', 9)

        # if mouse is clicked on the window close button, the program is stopped by setting current screen to quit
        if event.type == pygame.QUIT:
            current_screen = "quit"

    # If in manual mode and human has made a move, AI will make it
    if not autoMode:
        if turn == 1:
            moveAI()

    # If in auto mode, training will be done
    if autoMode and modePicked:
        train()

    # Plot a new point on the graph if it satisfies the condition
    if gameCounter - (multiplier * plotStep) == 0 and gameCounter > 0:
        plotGraph()
        multiplier = multiplier + 1


def plotGraph():
    global gameCounter
    global plotStep
    global multiplier
    global displayGraph

    global xAxis
    global yAxisWins
    global yAxisLoses
    global yAxisDraw

    global victoryNumAI
    global defeatNumAI
    global drawNum

    # Displaying graph's title
    plt.title('Graph of Game Result vs Number of Games')

    # Setting x and y axis ranges
    plt.xlim(0, gameCounter)
    plt.ylim(0, 100)

    # Restricting the scale of y-axis
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    # Adding x values to the list
    xAxis.append(plotStep * multiplier)

    # Calculating percentage values of wins, loses, and draws
    percentageWin = int((victoryNumAI/plotStep) * 100)
    percentageLose = int((defeatNumAI/plotStep) * 100)
    percentageDraw = int((drawNum/plotStep) * 100)

    # Adding y values to the list
    yAxisWins.append(percentageWin)
    yAxisLoses.append(percentageLose)
    yAxisDraw.append(percentageDraw)

    if displayGraph:
        # Plotting the graphs using the values from the lists
        plt.plot(xAxis, yAxisWins, color='red', linestyle='solid', linewidth=3)
        plt.plot(xAxis, yAxisLoses, color='blue', linestyle='solid', linewidth=3)
        plt.plot(xAxis, yAxisDraw, color='green', linestyle='solid', linewidth=3)

        # Converting integers to strings
        percentageWinsStr = str(percentageWin)
        percentageLosesStr = str(percentageLose)
        percentageDrawStr = str(percentageDraw)

        # Displaying graph's legend
        plt.legend(["Win - %" + percentageWinsStr, "Lose - %" + percentageLosesStr, "Draw - %" + percentageDrawStr], loc="upper left")

        # Saving graph
        plt.savefig('graph.png')

    # Setting variables to zero to calculate a new percentage value for next games
    victoryNumAI = 0
    defeatNumAI = 0
    drawNum = 0


def resultDetection():
    global gameCounter
    global moveCounter
    global gameAdded
    global current_episode
    global rewardHuman
    global rewardAI

    global victoryComb1
    global victoryComb2
    global victoryComb3
    global victoryComb4
    global victoryComb5
    global victoryComb6
    global victoryComb7
    global victoryComb8

    global charSquare1
    global charSquare2
    global charSquare3
    global charSquare4
    global charSquare5
    global charSquare6
    global charSquare7
    global charSquare8
    global charSquare9

    global victoryNumAI
    global defeatNumAI
    global drawNum

    # Setting variables for different rewards
    rewardVictory = 5
    rewardDefeat = -10
    rewardDraw = 10

    # These if statements are going through all the possible result combinations to check if one of them has occurred
    # Checking combinations for X
    if not gameAdded:
        # Horizontal combinations
        if charSquare1 == 'x' and charSquare2 == 'x' and charSquare3 == 'x':
            victoryComb1 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        elif charSquare4 == 'x' and charSquare5 == 'x' and charSquare6 == 'x':
            victoryComb2 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        elif charSquare7 == 'x' and charSquare8 == 'x' and charSquare9 == 'x':
            victoryComb3 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        # Vertical combinations
        elif charSquare1 == 'x' and charSquare4 == 'x' and charSquare7 == 'x':
            victoryComb4 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        elif charSquare2 == 'x' and charSquare5 == 'x' and charSquare8 == 'x':
            victoryComb5 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        elif charSquare3 == 'x' and charSquare6 == 'x' and charSquare9 == 'x':
            victoryComb6 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        # Diagonal combinations
        elif charSquare1 == 'x' and charSquare5 == 'x' and charSquare9 == 'x':
            victoryComb7 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        elif charSquare3 == 'x' and charSquare5 == 'x' and charSquare7 == 'x':
            victoryComb8 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardDefeat
            rewardHuman = rewardVictory
            defeatNumAI += 1
        # Checking combinations for O
        # Horizontal combinations
        elif charSquare1 == 'o' and charSquare2 == 'o' and charSquare3 == 'o':
            victoryComb1 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        elif charSquare4 == 'o' and charSquare5 == 'o' and charSquare6 == 'o':
            victoryComb2 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        elif charSquare7 == 'o' and charSquare8 == 'o' and charSquare9 == 'o':
            victoryComb3 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        # Vertical combinations
        elif charSquare1 == 'o' and charSquare4 == 'o' and charSquare7 == 'o':
            victoryComb4 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        elif charSquare2 == 'o' and charSquare5 == 'o' and charSquare8 == 'o':
            victoryComb5 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        elif charSquare3 == 'o' and charSquare6 == 'o' and charSquare9 == 'o':
            victoryComb6 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        # Diagonal combinations
        elif charSquare1 == 'o' and charSquare5 == 'o' and charSquare9 == 'o':
            victoryComb7 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        elif charSquare3 == 'o' and charSquare5 == 'o' and charSquare7 == 'o':
            victoryComb8 = True
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            rewardAI = rewardVictory
            rewardHuman = rewardDefeat
            victoryNumAI += 1
        elif moveCounter == 9:
            rewardAI = rewardDraw
            rewardHuman = rewardDraw
            gameCounter += 1
            current_episode += 1
            gameAdded = True
            drawNum += 1

    # Returns true if one of the combinations has been detected
    if rewardAI == 10 or rewardAI == 5 or rewardAI == -10:
        return True
    else:
        return False


def moveAI():
    global autoMode
    global stateID
    global turn

    global max_epsilon
    global min_epsilon
    global current_epsilonAI
    global rewardAI

    global old_q_valueAITemp
    global oldStateIDAITemp
    global action_indexAITemp
    global outputQvalues

    # Outputting the information below if true
    if outputQvalues:
        print("Current state's id:", getStateID(getCurrentState()))
        print('Total unique states:', stateID)
        print('Q values for current state:', q_valuesAI[getStateID(getCurrentState())])
        print('Current epsilon:', current_epsilonAI)

    # Checking if the result has not been detected
    if not resultDetection():
        # Choosing next action
        action_indexAI = getNextActionAI()

        # Saving id of the state before making a move
        oldStateIDAI = getStateID(getCurrentState())

        # Making a move based on an action chosen
        grid_displayCharacter(AICharacter, action_indexAI)

        # Updating epsilon
        current_epsilonAI = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * current_episode)

        # Getting current state's id
        currentStateID = getStateID(getCurrentState())

        # Calculating temporal difference
        old_q_valueAI = q_valuesAI[oldStateIDAI, action_indexAI]
        temporal_difference = rewardAI + (discount_factor * np.max(q_valuesAI[currentStateID])) - old_q_valueAI

        # Updating Q value for the previous state and action pair
        new_q_valueAI = old_q_valueAI + (learning_rate * temporal_difference)
        q_valuesAI[currentStateID, action_indexAI] = new_q_valueAI

        # Storing values for this state in temporary variables
        old_q_valueAITemp = old_q_valueAI
        oldStateIDAITemp = oldStateIDAI
        action_indexAITemp = action_indexAI

    if resultDetection():
        # Calculating temporal difference
        temporal_difference = rewardAI + (discount_factor * np.max(q_valuesAI[oldStateIDAITemp])) - old_q_valueAITemp

        # Updating Q value for this state based on the values from previous state and action pair
        new_q_valueAI = old_q_valueAITemp + (learning_rate * temporal_difference)
        q_valuesAI[oldStateIDAITemp, action_indexAITemp] = new_q_valueAI

    # Setting turn to zero
    turn = 0


def moveHuman(character, square):
    global turn

    # Displays character in a square picked by human
    if grid_displayCharacter(character, square) == 1:
        # Sets turn to 1 once the character has been displayed
        turn = 1


def getRandomMove(actions):
    # Returns a random move
    randomMove = random.choice(actions)
    return randomMove


def availablePositions():
    global usedS1
    global usedS2
    global usedS3
    global usedS4
    global usedS5
    global usedS6
    global usedS7
    global usedS8
    global usedS9

    # Initialising a list of possible actions
    actions = []

    # Adding square's number to the list if it is free
    if usedS1 == 0:
        actions.append(1)
    if usedS2 == 0:
        actions.append(2)
    if usedS3 == 0:
        actions.append(3)
    if usedS4 == 0:
        actions.append(4)
    if usedS5 == 0:
        actions.append(5)
    if usedS6 == 0:
        actions.append(6)
    if usedS7 == 0:
        actions.append(7)
    if usedS8 == 0:
        actions.append(8)
    if usedS9 == 0:
        actions.append(9)

    # Returns a list
    return actions


def grid_displayCharacter(character, square):
    global moveCounter

    global squareX1
    global squareX2
    global squareX3
    global squareX4
    global squareX5
    global squareX6
    global squareX7
    global squareX8
    global squareX9

    global squareO1
    global squareO2
    global squareO3
    global squareO4
    global squareO5
    global squareO6
    global squareO7
    global squareO8
    global squareO9

    global usedS1
    global usedS2
    global usedS3
    global usedS4
    global usedS5
    global usedS6
    global usedS7
    global usedS8
    global usedS9

    global charSquare1
    global charSquare2
    global charSquare3
    global charSquare4
    global charSquare5
    global charSquare6
    global charSquare7
    global charSquare8
    global charSquare9

    # Checking that the character passed as a parameter is x
    if character == 'x':
        # Checking that the square is not used and that is the that has been passed as a parameter
        if usedS1 == 0 and square == 1:
            # Setting squareX1 to True to display character x in the first square
            squareX1 = True
            # Setting this square as unavailable
            usedS1 = 1
            # Storing that character x is stored in this square
            charSquare1 = 'x'
            # Incrementing move counter
            moveCounter += 1
            # Returning 1 to show that the character has been displayed
            return 1
        elif usedS2 == 0 and square == 2:
            squareX2 = True
            usedS2 = 1
            charSquare2 = 'x'
            moveCounter += 1
            return 1
        elif usedS3 == 0 and square == 3:
            squareX3 = True
            usedS3 = 1
            charSquare3 = 'x'
            moveCounter += 1
            return 1
        elif usedS4 == 0 and square == 4:
            squareX4 = True
            usedS4 = 1
            charSquare4 = 'x'
            moveCounter += 1
            return 1
        elif usedS5 == 0 and square == 5:
            squareX5 = True
            usedS5 = 1
            charSquare5 = 'x'
            moveCounter += 1
            return 1
        elif usedS6 == 0 and square == 6:
            squareX6 = True
            usedS6 = 1
            charSquare6 = 'x'
            moveCounter += 1
            return 1
        elif usedS7 == 0 and square == 7:
            squareX7 = True
            usedS7 = 1
            charSquare7 = 'x'
            moveCounter += 1
            return 1
        elif usedS8 == 0 and square == 8:
            squareX8 = True
            usedS8 = 1
            charSquare8 = 'x'
            moveCounter += 1
            return 1
        elif usedS9 == 0 and square == 9:
            squareX9 = True
            usedS9 = 1
            charSquare9 = 'x'
            moveCounter += 1
            return 1

    elif character == 'o':
        if usedS1 == 0 and square == 1:
            squareO1 = True
            usedS1 = 1
            charSquare1 = 'o'
            moveCounter += 1
            return 1
        elif usedS2 == 0 and square == 2:
            squareO2 = True
            usedS2 = 1
            charSquare2 = 'o'
            moveCounter += 1
            return 1
        elif usedS3 == 0 and square == 3:
            squareO3 = True
            usedS3 = 1
            charSquare3 = 'o'
            moveCounter += 1
            return 1
        elif usedS4 == 0 and square == 4:
            squareO4 = True
            usedS4 = 1
            charSquare4 = 'o'
            moveCounter += 1
            return 1
        elif usedS5 == 0 and square == 5:
            squareO5 = True
            usedS5 = 1
            charSquare5 = 'o'
            moveCounter += 1
            return 1
        elif usedS6 == 0 and square == 6:
            squareO6 = True
            usedS6 = 1
            charSquare6 = 'o'
            moveCounter += 1
            return 1
        elif usedS7 == 0 and square == 7:
            squareO7 = True
            usedS7 = 1
            charSquare7 = 'o'
            moveCounter += 1
            return 1
        elif usedS8 == 0 and square == 8:
            squareO8 = True
            usedS8 = 1
            charSquare8 = 'o'
            moveCounter += 1
            return 1
        elif usedS9 == 0 and square == 9:
            squareO9 = True
            usedS9 = 1
            charSquare9 = 'o'
            moveCounter += 1
            return 1


def getCurrentState():
    global usedS1
    global usedS2
    global usedS3
    global usedS4
    global usedS5
    global usedS6
    global usedS7
    global usedS8
    global usedS9

    global squareX1
    global squareX2
    global squareX3
    global squareX4
    global squareX5
    global squareX6
    global squareX7
    global squareX8
    global squareX9

    # String that stores current state as a set of numbers
    currentState = ''

    # If the cell is not used, then adding 0 to the string
    if usedS1 == 0:
        currentState = currentState + '0'
    # If the cell is used by character X, then adding 1 to the string
    elif squareX1:
        currentState = currentState + '1'
    # The only option left is that if the cell stores O, so adding 2 to the string
    else:
        currentState = currentState + '2'
    if usedS2 == 0:
        currentState = currentState + '0'
    elif squareX2:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'
    if usedS3 == 0:
        currentState = currentState + '0'
    elif squareX3:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'
    if usedS4 == 0:
        currentState = currentState + '0'
    elif squareX4:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'
    if usedS5 == 0:
        currentState = currentState + '0'
    elif squareX5:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'
    if usedS6 == 0:
        currentState = currentState + '0'
    elif squareX6:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'
    if usedS7 == 0:
        currentState = currentState + '0'
    elif squareX7:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'
    if usedS8 == 0:
        currentState = currentState + '0'
    elif squareX8:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'
    if usedS9 == 0:
        currentState = currentState + '0'
    elif squareX9:
        currentState = currentState + '1'
    else:
        currentState = currentState + '2'

    # Returning string storing current state
    return currentState


def getStateID(state):
    global statesDict
    global stateID

    # Checking if the state passed as a parameter exists
    for i in range(stateID + 1):
        # If it exists, then returning its id
        if statesDict.get(i) == state:
            return i
    # If it does not exist, then assigning it to a new id and returning it
    else:
        stateID = stateID + 1
        statesDict[stateID] = state
        return stateID


def getNextActionAI():
    global current_epsilonAI
    global moveCounter

    # Gets a random value between 0.0 and 1.0
    randomNumber = random.random()
    # If it is greater than epsilon, then it will make an action which has the highest q value for current state (exploit)
    if randomNumber > current_epsilonAI:
        currentStateID = getStateID(getCurrentState())
        move = np.argmax(q_valuesAI[currentStateID])

        # If move is 0 (means that all the values for current state are zero because AI has not developed enough yet), then get a random move
        if move == 0 and availablePositions():
            # Returning random move
            return getRandomMove(availablePositions())
        else:
            # Returning move with the highest q value
            return move

    # If it is less than epsilon, then it will make a random action
    else:
        # If there are no positions available it set move counter to 9 to stop the game
        if not availablePositions():
            moveCounter = 9
        else:
            # Returning random move
            return getRandomMove(availablePositions())


def getNextActionHuman():
    global current_epsilonHuman
    global moveCounter

    # Gets a random value between 0.0 and 1.0
    randomNumber = random.random()
    # If it is greater than epsilon, then it will make an action which has the highest q value for current state (exploit)
    if randomNumber > current_epsilonHuman:
        currentStateID = getStateID(getCurrentState())
        move = np.argmax(q_valuesHuman[currentStateID])

        # If move is 0 (means that all the values for current state are zero because AI has not developed enough yet), then explore
        if move == 0 and availablePositions():
            # Returning random move
            return getRandomMove(availablePositions())
        else:
            # Returning move with the highest q value
            return move

    # If it is less than epsilon, then it will make a random action (explore)
    else:
        # If there are no positions available it set move counter to 9 to stop the game
        if not availablePositions():
            moveCounter = 9
        else:
            # Returning random move
            return getRandomMove(availablePositions())


def clearBoard():
    global moveCounter
    global displayWarning
    global gameAdded
    global turn

    global rewardHuman
    global rewardAI

    global squareX1
    global squareX2
    global squareX3
    global squareX4
    global squareX5
    global squareX6
    global squareX7
    global squareX8
    global squareX9
    global squareO1
    global squareO2
    global squareO3
    global squareO4
    global squareO5
    global squareO6
    global squareO7
    global squareO8
    global squareO9

    global charSquare1
    global charSquare2
    global charSquare3
    global charSquare4
    global charSquare5
    global charSquare6
    global charSquare7
    global charSquare8
    global charSquare9

    global victoryComb1
    global victoryComb2
    global victoryComb3
    global victoryComb4
    global victoryComb5
    global victoryComb6
    global victoryComb7
    global victoryComb8

    global usedS1
    global usedS2
    global usedS3
    global usedS4
    global usedS5
    global usedS6
    global usedS7
    global usedS8
    global usedS9

    # Resetting certain variables upon finishing the game
    squareX1 = False
    squareX2 = False
    squareX3 = False
    squareX4 = False
    squareX5 = False
    squareX6 = False
    squareX7 = False
    squareX8 = False
    squareX9 = False

    squareO1 = False
    squareO2 = False
    squareO3 = False
    squareO4 = False
    squareO5 = False
    squareO6 = False
    squareO7 = False
    squareO8 = False
    squareO9 = False

    victoryComb1 = False
    victoryComb2 = False
    victoryComb3 = False
    victoryComb4 = False
    victoryComb5 = False
    victoryComb6 = False
    victoryComb7 = False
    victoryComb8 = False

    charSquare1 = ''
    charSquare2 = ''
    charSquare3 = ''
    charSquare4 = ''
    charSquare5 = ''
    charSquare6 = ''
    charSquare7 = ''
    charSquare8 = ''
    charSquare9 = ''

    usedS1 = 0
    usedS2 = 0
    usedS3 = 0
    usedS4 = 0
    usedS5 = 0
    usedS6 = 0
    usedS7 = 0
    usedS8 = 0
    usedS9 = 0

    rewardAI = 0
    rewardHuman = 0

    moveCounter = 0
    turn = 0

    gameAdded = False
    displayWarning = False


def gameTextLocation(gameCounter):
    global gameXcoord
    global gameYcoord

    # Changing game counter's location based on its value
    if gameCounter < 10:
        gameXcoord = 698
        gameYcoord = 310
    elif 9 < gameCounter < 100:
        gameXcoord = 690
        gameYcoord = 310
    elif 99 < gameCounter < 1000:
        gameXcoord = 680
        gameYcoord = 310
    elif 999 < gameCounter < 10000:
        gameXcoord = 680
        gameYcoord = 310
    elif 9999 < gameCounter < 100000:
        gameXcoord = 670
        gameYcoord = 310
    elif 99999 < gameCounter < 1000000:
        gameXcoord = 660
        gameYcoord = 310
    elif 999999 < gameCounter < 10000000:
        gameXcoord = 650
        gameYcoord = 310


def timeElapsed():
    global start_time

    # Calculating elapsed time using current time and the time when the game was opened
    elapsed_time = time.time() - start_time
    # Converting time to minutes
    minutes, seconds = divmod(elapsed_time, 60)
    # Converting time to hours
    hours, minutes = divmod(minutes, 60)
    # Storing time as a single string
    time_string = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    # Returning time elapsed
    return time_string


def grid():
    global grid_width
    global grid_height
    global grid_spacing
    global grid_firstPosX
    global grid_firstPosY

    # Declaring grid-related variable
    grid_lineWidthV = grid_spacing
    grid_lineHeightV = grid_spacing * 2 + grid_height * 3
    grid_lineWidthH = grid_spacing * 2 + grid_width * 3
    grid_lineHeightH = grid_spacing
    grid_lineRadius = 8

    # Displaying grid
    # Displaying row 1
    pygame.draw.rect(screen, colour_white, [grid_firstPosX, grid_firstPosY, grid_width, grid_height])
    pygame.draw.rect(screen, colour_white, [grid_firstPosX + grid_width + grid_spacing, grid_firstPosY, grid_width, grid_height])
    pygame.draw.rect(screen, colour_white, [grid_firstPosX + (grid_width + grid_spacing) * 2, grid_firstPosY, grid_width, grid_height])
    # Displaying row 2
    pygame.draw.rect(screen, colour_white, [grid_firstPosX, grid_firstPosY + grid_spacing + grid_height, grid_width, grid_height])
    pygame.draw.rect(screen, colour_white, [grid_firstPosX + grid_width + grid_spacing, grid_firstPosY + grid_spacing + grid_height, grid_width, grid_height])
    pygame.draw.rect(screen, colour_white, [grid_firstPosX + (grid_width + grid_spacing) * 2, grid_firstPosY + grid_spacing + grid_height, grid_width, grid_height])
    # Displaying row 3
    pygame.draw.rect(screen, colour_white, [grid_firstPosX, grid_firstPosY + (grid_spacing + grid_height) * 2, grid_width, grid_height])
    pygame.draw.rect(screen, colour_white, [grid_firstPosX + grid_width + grid_spacing, grid_firstPosY + (grid_spacing + grid_height) * 2, grid_width, grid_height])
    pygame.draw.rect(screen, colour_white, [grid_firstPosX + (grid_width + grid_spacing) * 2, grid_firstPosY + (grid_spacing + grid_height) * 2, grid_width, grid_height])
    # Displaying separating vertical lines
    pygame.draw.rect(screen, colour_redLight, [grid_firstPosX + grid_width, grid_firstPosY, grid_lineWidthV, grid_lineHeightV], 0, 0, grid_lineRadius, grid_lineRadius, grid_lineRadius, grid_lineRadius)
    pygame.draw.rect(screen, colour_redLight, [grid_firstPosX + grid_width * 2 + grid_spacing, grid_firstPosY, grid_lineWidthV, grid_lineHeightV], 0, 0, grid_lineRadius, grid_lineRadius, grid_lineRadius, grid_lineRadius)
    # Displaying separating horizontal lines
    pygame.draw.rect(screen, colour_redLight, [grid_firstPosX, grid_firstPosY + grid_width, grid_lineWidthH, grid_lineHeightH], 0, 0, grid_lineRadius, grid_lineRadius, grid_lineRadius, grid_lineRadius)
    pygame.draw.rect(screen, colour_redLight, [grid_firstPosX, grid_firstPosY + grid_height * 2 + grid_spacing, grid_lineWidthH, grid_lineHeightH], 0, 0, grid_lineRadius, grid_lineRadius, grid_lineRadius, grid_lineRadius)


def train():
    global current_episode
    global rewardHuman
    global rewardAI

    global max_epsilon
    global min_epsilon
    global current_epsilonHuman
    global current_epsilonAI

    global old_q_valueHumanTemp
    global oldStateIDHumanTemp
    global action_indexHumanTemp
    global old_q_valueAITemp
    global oldStateIDAITemp
    global action_indexAITemp

    # Checking if the result has not been detected
    if not resultDetection():
        # Making an automatic move for human
        # Choosing next action
        action_indexHuman = getNextActionHuman()

        # Saving id of the state before making a move
        oldStateIDHuman = getStateID(getCurrentState())

        # Making a move based on an action chosen
        grid_displayCharacter(humanCharacter, action_indexHuman)

        # Updating epsilon
        current_epsilonHuman = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * current_episode)

        # Getting current state's id
        currentStateID = getStateID(getCurrentState())

        # Calculating temporal difference
        old_q_valueHuman = q_valuesHuman[oldStateIDHuman, action_indexHuman]
        temporal_difference = rewardHuman + (discount_factor * np.max(q_valuesHuman[currentStateID])) - old_q_valueHuman

        # Updating Q value for the previous state and action pair
        new_q_valueHuman = old_q_valueHuman + (learning_rate * temporal_difference)
        q_valuesHuman[currentStateID, action_indexHuman] = new_q_valueHuman

        # Storing values for this state in temporary variables
        old_q_valueHumanTemp = old_q_valueHuman
        oldStateIDHumanTemp = oldStateIDHuman
        action_indexHumanTemp = action_indexHuman

    # Checking if the result has not been detected
    if not resultDetection():
        # Making an automatic move for AI
        # Choosing next action
        action_indexAI = getNextActionAI()

        # Saving id of the state before making a move
        oldStateIDAI = getStateID(getCurrentState())

        # Making a move based on an action chosen
        grid_displayCharacter(AICharacter, action_indexAI)

        # Updating epsilon
        current_epsilonAI = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * current_episode)

        # Getting current state's id
        currentStateID = getStateID(getCurrentState())

        # Calculating temporal difference
        old_q_valueAI = q_valuesAI[oldStateIDAI, action_indexAI]
        temporal_difference = rewardAI + (discount_factor * np.max(q_valuesAI[currentStateID])) - old_q_valueAI

        # Updating Q value for the previous state and action pair
        new_q_valueAI = old_q_valueAI + (learning_rate * temporal_difference)
        q_valuesAI[currentStateID, action_indexAI] = new_q_valueAI

        # Storing values for this state in temporary variables
        old_q_valueAITemp = old_q_valueAI
        oldStateIDAITemp = oldStateIDAI
        action_indexAITemp = action_indexAI

    # Checking if the result has been detected
    if resultDetection():
        # Calculating temporal difference for Human
        temporal_difference = rewardHuman + (discount_factor * np.max(q_valuesHuman[oldStateIDHumanTemp])) - old_q_valueHumanTemp

        # Updating Q value for this state based on the values from previous state and action pair
        new_q_valueHuman = old_q_valueHumanTemp + (learning_rate * temporal_difference)
        q_valuesHuman[oldStateIDHumanTemp, action_indexHumanTemp] = new_q_valueHuman

        # Calculating temporal difference for AI
        temporal_difference = rewardAI + (discount_factor * np.max(q_valuesAI[oldStateIDAITemp])) - old_q_valueAITemp

        # Updating Q value for this state based on the values from previous state and action pair
        new_q_valueAI = old_q_valueAITemp + (learning_rate * temporal_difference)
        q_valuesAI[oldStateIDAITemp, action_indexAITemp] = new_q_valueAI

        # Clearing the board
        clearBoard()


# Main loop that keeps the program running
while running:
    # Constantly checking if the result has been detected
    resultDetection()

    # Constantly updating coordinates of the mouse
    mouse = pygame.mouse.get_pos()

    # Displaying needed screen
    if current_screen == "menu":
        menu_screen()
    elif current_screen == "about":
        about_screen()
    elif current_screen == "settings":
        settings_screen()
    elif current_screen == "game":
        game_screen()

        # Displaying character image in a certain square if required
        if squareX1:
            screen.blit(imgX, (grid_firstPosX, grid_firstPosY))
        if squareX2:
            screen.blit(imgX, (grid_firstPosX + grid_width + grid_spacing, grid_firstPosY))
        if squareX3:
            screen.blit(imgX, (grid_firstPosX + 2*grid_width + 2*grid_spacing, grid_firstPosY))
        if squareX4:
            screen.blit(imgX, (grid_firstPosX, grid_firstPosY + grid_height + grid_spacing))
        if squareX5:
            screen.blit(imgX, (grid_firstPosX + grid_width + grid_spacing, grid_firstPosY + grid_height + grid_spacing))
        if squareX6:
            screen.blit(imgX, (grid_firstPosX + 2*grid_width + 2*grid_spacing, grid_firstPosY + grid_height + grid_spacing))
        if squareX7:
            screen.blit(imgX, (grid_firstPosX, grid_firstPosY + 2*grid_height + 2*grid_spacing))
        if squareX8:
            screen.blit(imgX, (grid_firstPosX + grid_width + grid_spacing, grid_firstPosY + 2*grid_height + 2*grid_spacing))
        if squareX9:
            screen.blit(imgX, (grid_firstPosX + 2*grid_width + 2*grid_spacing, grid_firstPosY + 2*grid_height + 2*grid_spacing))
        if squareO1:
            screen.blit(imgO, (grid_firstPosX, grid_firstPosY))
        if squareO2:
            screen.blit(imgO, (grid_firstPosX + grid_width + grid_spacing, grid_firstPosY))
        if squareO3:
            screen.blit(imgO, (grid_firstPosX + 2 * grid_width + 2 * grid_spacing, grid_firstPosY))
        if squareO4:
            screen.blit(imgO, (grid_firstPosX, grid_firstPosY + grid_height + grid_spacing))
        if squareO5:
            screen.blit(imgO, (grid_firstPosX + grid_width + grid_spacing, grid_firstPosY + grid_height + grid_spacing))
        if squareO6:
            screen.blit(imgO, (grid_firstPosX + 2 * grid_width + 2 * grid_spacing, grid_firstPosY + grid_height + grid_spacing))
        if squareO7:
            screen.blit(imgO, (grid_firstPosX, grid_firstPosY + 2 * grid_height + 2 * grid_spacing))
        if squareO8:
            screen.blit(imgO, (grid_firstPosX + grid_width + grid_spacing, grid_firstPosY + 2 * grid_height + 2 * grid_spacing))
        if squareO9:
            screen.blit(imgO, (grid_firstPosX + 2 * grid_width + 2 * grid_spacing, grid_firstPosY + 2 * grid_height + 2 * grid_spacing))

        # Displaying certain victory combination line image if required
        if victoryComb1:
            screen.blit(lineHorizontal, (grid_firstPosX, grid_firstPosY+55))
        elif victoryComb2:
            screen.blit(lineHorizontal, (grid_firstPosX, grid_firstPosY + grid_height + grid_spacing+55))
        elif victoryComb3:
            screen.blit(lineHorizontal, (grid_firstPosX, grid_firstPosY + 2 * grid_height + 2 * grid_spacing+55))
        elif victoryComb4:
            screen.blit(lineVertical, (grid_firstPosX+59, grid_firstPosY-3))
        elif victoryComb5:
            screen.blit(lineVertical, (grid_firstPosX + grid_width + grid_spacing+59, grid_firstPosY-3))
        elif victoryComb6:
            screen.blit(lineVertical, (grid_firstPosX + 2 * grid_width + 2 * grid_spacing+59, grid_firstPosY-3))
        elif victoryComb7:
            screen.blit(lineDiagonal1, (grid_firstPosX+2, grid_firstPosY))
        elif victoryComb8:
            screen.blit(lineDiagonal2, (grid_firstPosX-7, grid_firstPosY))

        if displayGraph:
            # Loading the graph from the folder as it changes while the program runs
            graph = pygame.image.load("/Users/dima/Desktop/DmytroDudarenko22-Ben-Coursework/Coursework Code/Code/graph.png").convert()
            # Displaying the graph
            screen.blit(graph, (graphXcoord, graphYcoord))

    elif current_screen == "quit":
        running = False

    for event in pygame.event.get():
        # if mouse is clicked on the window close button, the program is stopped
        if event.type == pygame.QUIT:
            running = False

    # Updating the window
    pygame.display.update()
