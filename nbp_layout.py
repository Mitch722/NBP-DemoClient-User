import argparse
import atexit
import locale
import platform
import threading
import time
import tkinter as tk
from tkinter import E, N, S, W

import requests

import npr_move

__version__ = '0.1.2' # Major, minor, patch

# Platforms
WINDOWS = (platform.system() == 'Windows')
LINUX = (platform.system() == 'Linux')
MAC = (platform.system() == 'Darwin')

BASE_URL = 'https://rmm8lvjage.execute-api.eu-west-1.amazonaws.com/prod/' # Test server
APP_TITLE = 'Number Plate Recognition Game client - v' + __version__
GET_LIST_OF_GAME_STYLES_EXTENSION = 'GM-GetListOfGameStyles'
OFFER_GAME_EXTENSION = 'GM-OfferGame'
POLL_FOR_GAME_STATE_EXTENSION = 'GM-PollForGameState'
MAKE_MOVE_EXTENSION = 'GM-MakeMove'
CANCEL_GAME_OFFER_EXTENSION = 'GM-CancelGameOffer'
CANCEL_GAME_TEXT = 'Cancel Game'

API_CALL_HEADERS = {'Content-Type': 'application/json'}

GAME_TYPE_ID = 55 # Number plates

GAME_STYLE_LISTBOX_ACTIVE_STYLE = 'none'

GAME_STYLE_LISTBOX_DEFAULT_SELECTION = 0

DISABLED = 'disable'
ENABLED = 'normal'


class NPRDemoClient(tk.Frame):
    def __init__(self, root, args):
        tk.Frame.__init__(self, root)
        locale.setlocale(locale.LC_ALL, '')  # empty string for platform's default settings
        self.master = root
        root.title(APP_TITLE)
        root.resizable(False, False)
        try:
            root.iconbitmap('./200x200/icon.ico')
        except Exception as e:
            print(e)
        atexit.register(self.cancel_game)

        # Init class data fields that we use for storing info that we need for using the API
        self.bot_id = None
        self.bot_password = None
        self.logged_in = False
        self.game_style_ids = []
        self.gameChips = 0
        self.gameDeals = 0
        self.gameStake = 0
        self.gamePrize = 0
        self.player_key = None
        self.play_again = tk.BooleanVar()
        self.do_not_play_same_user = tk.BooleanVar()
        self.close_after_game = False
        self.game_cancelled = False
        self.in_game = False

        self.topFrame = tk.Frame(root, padx=12, pady=12)
        self.middleFrame = tk.Frame(root, padx=12)
        self.middleFrameLeft = tk.Frame(self.middleFrame)
        self.middleFrameRight = tk.Frame(self.middleFrame)
        self.middleFrameRighter = tk.Frame(self.middleFrame)

        self.topFrame.grid(row=0, sticky=W + E)

        self.middleFrame.grid(row=1, sticky=W)
        self.middleFrameLeft.grid(row=1, column=0)
        self.middleFrameRight.grid(row=1, column=1)
        self.middleFrameRighter.grid(row=1, column=2)

        # ===================================
        # Create form elements

        # Top Frame Elements
        self.botNameLabel = tk.Label(self.topFrame, text='Bot Name:')
        self.bot_id_entry = tk.Entry(self.topFrame)
        self.bot_id_entry.bind('<Return>', self.log_in_if_not)
        self.bot_id_entry.focus()
        self.passwordLabel = tk.Label(self.topFrame, text='Password:')
        self.bot_password_entry = tk.Entry(self.topFrame, show='*')
        self.bot_password_entry.bind('<Return>', self.log_in_if_not)
        self.log_in_out_button = tk.Button(self.topFrame, text='Login', command=self.log_in_out_clicked)

        self.balanceLabel = tk.Label(self.topFrame, text='Bot Balance:')
        self.balance = tk.Label(self.topFrame, text='0')
        self.close_button = tk.Button(self.topFrame, text='Close', padx=2, command=root.destroy)

        # Middle Frame Elements
        # Middle Frame LEFT Elements
        self.gameStyleLabel = tk.Label(self.middleFrameLeft, font=(None, 18), pady=0, text='Game Style Selection')

        self.opponentLabel = tk.Label(self.middleFrameLeft, text='Specify Opponent (optional):')
        self.specify_opponent_entry = tk.Entry(self.middleFrameLeft)

        self.do_not_play_same_user_check = tk.Checkbutton(self.middleFrameLeft,
                                                       text='Don\'t play another bot in same user account as me',
                                                       var=self.do_not_play_same_user)

        self.game_styles_listbox = tk.Listbox(self.middleFrameLeft, background='#FFFFFF', height=8)
        self.game_styles_listbox.bind('<Double-1>', self.find_game_double_clicked)
        self.game_styles_listbox.bind('<Return>',
                                      self.find_game_double_clicked)  # Not a double click but we want it to do the same thing

        self.refresh_game_styles_button = tk.Button(self.middleFrameLeft, text='Refresh Game Styles',
                                                 command=self.refresh_game_styles_clicked)

        self.thinkingTimeLabel = tk.Label(self.middleFrameLeft, text='Add \'Thinking Time\' (ms):')
        self.thinking_time_entry = tk.Entry(self.middleFrameLeft)

        self.auto_play_next_game_check = tk.Checkbutton(self.middleFrameLeft, text='Play another game when complete',
                                                     var=self.play_again)

        self.cancel_stop_game_button = tk.Button(self.middleFrameLeft, text=CANCEL_GAME_TEXT,
                                              command=self.cancel_stop_game_clicked)
        self.find_game_button = tk.Button(self.middleFrameLeft, text='Find Game', command=self.find_game_clicked)

        self.resultText = tk.Message(self.middleFrameLeft, width=500,
                                  text='This is where the informational messages will appear')
        self.spacerLabel = tk.Label(self.middleFrameLeft, text=' ')

        # Middle Frame RIGHT Elements

        self.gameTitleLabel = tk.Label(self.middleFrameRight, text='Game Title')
        self.gameTitleText = tk.Text(self.middleFrameRight, height=3, background='white', spacing1=3, pady=0)

        # self.player = battleships_visuals.BattleshipsVisuals(self.middleFrameRight)  # Game Display Table
        # self.opponent = battleships_visuals.BattleshipsVisuals(self.middleFrameRight)  # Game Display Table
        self.gameActionLabel = tk.Label(self.middleFrameRight, text='')

        # ===================================
        # Set initial element states

        self.set_gamestyle_controls_states(DISABLED)
        self.cancel_stop_game_button.config(state=DISABLED)
        self.game_styles_listbox.config(background='white')
        self.thinking_time_entry.insert(0, 100)
        self.gameTitleText.config(state=DISABLED)
        self.set_balance(0)
        self.gameTitleText.tag_configure('center', justify='center')
        self.gameTitleText.tag_configure('bold', font='-weight bold')

        # ===================================
        # Form Layout

        # Top Frame Form Layout
        self.topFrame.grid_rowconfigure(0, weight=1)
        self.botNameLabel.grid(row=0, column=0, sticky=E)
        self.bot_id_entry.grid(row=0, column=1, sticky=W)
        self.passwordLabel.grid(row=0, column=2, sticky=E)
        self.bot_password_entry.grid(row=0, column=3, sticky=W)
        self.log_in_out_button.grid(row=0, column=4, sticky=E)
        self.topFrame.grid_columnconfigure(5, weight=1)
        self.balanceLabel.grid(row=0, column=5, sticky=E)
        self.balance.grid(row=0, column=6, sticky=W)
        self.close_button.grid(row=0, column=7, sticky=E, padx=(50, 0))

        # Middle Frame Form Layout
        self.middleFrame.grid_rowconfigure(0, weight=1)
        self.gameStyleLabel.grid(row=0, column=0, columnspan=1, sticky=W + E)
        self.spacerLabel.grid(row=0, column=2, sticky=E)

        self.opponentLabel.grid(row=2, column=0, sticky=W, pady=4)
        self.specify_opponent_entry.grid(row=2, column=0, sticky=E, pady=4)

        self.do_not_play_same_user_check.grid(row=3, column=0, columnspan=1, sticky='we', pady=4)
        self.game_styles_listbox.grid(row=4, column=0, columnspan=1, sticky='we', pady=4)
        self.find_game_button.grid(row=5, column=0, pady=4, sticky=W)
        self.refresh_game_styles_button.grid(row=5, column=0, columnspan=1, sticky='', pady=4)
        self.cancel_stop_game_button.grid(row=5, column=0, sticky=E)

        self.thinkingTimeLabel.grid(row=6, column=0, sticky=W, pady=4)
        self.thinking_time_entry.grid(row=6, column=0, sticky=E, pady=4)

        self.auto_play_next_game_check.grid(row=7, column=0, columnspan=1, sticky=W, pady=4)
        self.resultText.grid(row=9, column=0, columnspan=2, sticky=W, pady=4)
        self.middleFrame.grid_columnconfigure(9, weight=1)

        self.gameTitleLabel.grid(row=0, column=3)
        self.gameTitleText.grid(row=0, column=3, columnspan=2)
        # self.player.grid(row=1, column=3)
        # self.opponent.grid(row=1, column=4)
        self.gameActionLabel.grid(row=11, column=3, sticky='w')

        # Autoplay/autologin

        if args.autoplay and args.botid is not None:
            self.auto_play(args)
        elif args.botid is not None:
            self.bot_id_entry.insert(0, args.botid)
            self.bot_password_entry.insert(0, args.password)
            self.log_in_out_clicked()

    def auto_play(self, args):
        self.bot_id_entry.insert(0, args.botid)
        self.bot_password_entry.insert(0, args.password)
        self.log_in_out_clicked()
        self.thinking_time_entry.insert(0, args.timeout)
        if args.playanothergame:
            self.auto_play_next_game_check.select()
        if args.dontplaysameuserbot:
            self.do_not_play_same_user_check.select()
        if args.closeaftergame:
            self.close_after_game = True
        i = 0
        for i in range(self.game_styles_listbox.size()):
            if args.gamestyle in str(self.game_styles_listbox.get(i)):
                break
        self.game_styles_listbox.select_set(i, i)
        self.find_game_clicked()

    def log_in_out_clicked(self):
        ''' Click handler for the 'Login'/'Logout' button '''

        if self.logged_in: # Logging in
            self.resultText.config(text='Logged Out')

            self.master.title(APP_TITLE + ' (Not Logged In)')

            self.cancel_game()

            self.bot_id = None
            self.bot_password = None
            self.clear_game_title_text()
            self.gameActionLabel.config(text='')
            self.reset_game_styles_listbox()
            # self.clear_all_boards()
            # self.opponent.delete('all')

            self.log_in_out_button.config(text='Login')

            self.set_login_controls_states(ENABLED)
            self.set_gamestyle_controls_states(DISABLED)

            self.logged_in = False
            self.bot_password_entry.delete(0, 'end')
            self.set_balance(0)

        else: # Logging out
            self.bot_id = self.bot_id_entry.get()
            self.bot_password = self.bot_password_entry.get()

            res = self.get_list_of_game_styles()
            if res['Result'] == 'SUCCESS':
                self.resultText.config(text='Logged In')

                game_styles = res['GameStyles']
                self.master.title(self.bot_id + ' - ' + APP_TITLE)

                self.set_login_controls_states(DISABLED)
                self.set_gamestyle_controls_states(ENABLED)

                self.set_game_styles_listbox(game_styles)
                self.set_balance(res['Balance'])
                self.log_in_out_button.config(text='Logout')
                self.logged_in = True

            else:
                tk.messagebox.showerror('Error', 'Invalid login attempt. Please check the username and password entered.')

    def log_in_if_not(self, _):
        ''' Log in if not yet logged in '''
        if not self.logged_in: self.log_in_out_clicked()

    # def clear_all_boards(self):
    #     self.player.delete('all')
    #     self.opponent.delete('all')
    #     self.player.myBoard = None
    #     self.opponent.oppBoard = None

    def set_in_game(self, value):
        self.in_game = value

    def set_game_title_text(self, text, tag):
        self.gameTitleText.config(state=ENABLED)
        self.gameTitleText.insert('end', text, ('center', tag))
        self.gameTitleText.config(state=DISABLED)

    def clear_game_title_text(self):
        self.gameTitleText.config(state=ENABLED)
        self.gameTitleText.delete('1.0', 'end')
        self.gameTitleText.config(state=DISABLED)

    def set_login_controls_states(self, state):
        self.bot_id_entry.config(state=state)
        self.bot_password_entry.config(state=state)

    def set_gamestyle_controls_states(self, state):
        self.specify_opponent_entry.config(state=state)
        self.do_not_play_same_user_check.config(state=state)
        self.game_styles_listbox.config(state=state)
        self.find_game_button.config(state=state)
        self.refresh_game_styles_button.config(state=state)
        self.auto_play_next_game_check.config(state=state)
        self.thinking_time_entry.config(state=state)
        self.opponentLabel.config(state=state)
        self.thinkingTimeLabel.config(state=state)
        self.balanceLabel.config(state=state)
        self.balance.config(state=state)
        self.gameStyleLabel.config(state=state)
        self.game_styles_listbox.config(state=state)
        # self.player.config(state=state)
        # self.opponent.config(state=state)

    def set_balance(self, balance):
        ''' Set the balance field '''
        self.balance['text'] = int_with_commas(balance)
        self.balance['text'] += ' sat'

    def get_list_of_game_styles(self):
        ''' Get list of game styles from the server '''

        req = {'BotId': self.bot_id,
               'BotPassword': self.bot_password,
               'GameTypeId': GAME_TYPE_ID}

        url = BASE_URL + GET_LIST_OF_GAME_STYLES_EXTENSION

        return NPRDemoClient.make_api_call(url, req)

    def set_game_styles_listbox(self, game_styles: 'list of GameStyle dictionaries'):
        ''' Set the content of the game styles listbox with a list of GameStyle dictionaries
        game_styles are obtained through get_list_of_game_styles()
        '''
        GAME_STYLE_LISTBOX_TEXT = '{0:d} | {1:d} sat | {2:d} plates | {3:d}ms'
        # GAME_STYLE_LISTBOX_TEXT_BUFFER = 4
        # GAME_STYLE_LISTBOX_TEXT_LEN = len(GAME_STYLE_LISTBOX_TEXT) + GAME_STYLE_LISTBOX_TEXT_BUFFER
        self.reset_game_styles_listbox()
        for index, game_style in enumerate(game_styles):
            line = GAME_STYLE_LISTBOX_TEXT.format(game_style['GameStyleId'],
                                                  game_style['Stake'],
                                                  game_style['GameTypeSpecificInfo']['NumNPlates'],
                                                  game_style['GameTypeSpecificInfo']['GameLength'])
            self.game_styles_listbox.insert(index, line)
            self.game_style_ids.append(game_style['GameStyleId'])

            # self.game_styles_listbox.select_set(GAME_STYLE_LISTBOX_DEFAULT_SELECTION)

    def reset_game_styles_listbox(self):
        ''' Clear the content of the game styles listbox '''
        if self.game_styles_listbox.size() != 0:
            self.game_styles_listbox.delete(0, 'end')
            self.game_style_ids = []

    def refresh_game_styles_clicked(self):
        ''' Click handler for the 'Refresh Game Styles' button '''
        res = self.get_list_of_game_styles()
        game_styles = res['GameStyles']
        self.set_game_styles_listbox(game_styles)

    def find_game_clicked(self):
        ''' Click handler for the 'Find Game' button '''
        self.find_game_button.config(state=DISABLED)
        self.cancel_stop_game_button.config(state=ENABLED)
        # self.clear_all_boards()
        if not MAC: # Dispatch the work to a separate thread, to keep the GUI responsive
            threading.Thread(target=self.game_loop, daemon=True).start()
        else: # Doesn't work on MACs
            self.game_loop()

    def find_game_double_clicked(self, _):
        self.find_game_clicked()

    def game_loop(self):
        ''' Loop through finding and playing games '''
        while True:
            # self.clear_all_boards()
            self.find_game()
            if self.game_cancelled: break
            self.play_game()
            if self.close_after_game: self.close_button.invoke()
            if self.game_cancelled: break
            if not self.play_again.get(): break

        self.find_game_button.config(state=ENABLED)
        self.cancel_stop_game_button.config(state=DISABLED, text=CANCEL_GAME_TEXT)
        self.game_cancelled = False

    def find_game(self):
        ''' Find a game '''
        offer_game_res = self.offer_game()

        if offer_game_res['Result'] == 'INVALID_LOGIN_OR_PASSWORD':
            self.cancel_stop_game_clicked()
            if 'ErrorMessage' in offer_game_res and offer_game_res['ErrorMessage'] == 'Check of OpponentId failed':
                self.resultText.config(text='Invalid Opponent ID')
            else:
                self.resultText.config(text='Invalid login or password')
        elif offer_game_res['Result'] == 'INSUFFICIENT_BALANCE':
            self.cancel_stop_game_clicked()
            self.resultText.config(text='Insufficient balance')
        elif offer_game_res['Result'] == 'BOT_IS_INACTIVE':
            self.cancel_stop_game_clicked()
            self.resultText.config(text='Bot is inactive')
        else:
            self.player_key = offer_game_res['PlayerKey']
            if offer_game_res['Result'] == 'WAITING_FOR_GAME':
                self.wait_for_game()

    def offer_game(self):
        ''' Offer a game '''

        opponent_id = self.specify_opponent_entry.get()
        if len(opponent_id) == 0:
            opponent_id = None
        try:
            game_style_id = self.game_style_ids[int(self.game_styles_listbox.curselection()[0])]
        except IndexError:
            self.game_styles_listbox.select_set(GAME_STYLE_LISTBOX_DEFAULT_SELECTION)
            game_style_id = self.game_style_ids[0]

        req = {'BotId': self.bot_id,
               'BotPassword': self.bot_password,
               'MaximumWaitTime': 1000,
               'GameStyleId': game_style_id,
               'DontPlayAgainstSameUser': self.do_not_play_same_user.get(),
               'DontPlayAgainstSameBot': False,
               'OpponentId': opponent_id}
        url = BASE_URL + OFFER_GAME_EXTENSION

        return NPRDemoClient.make_api_call(url, req)

    def wait_for_game(self):
        '''Wait for game to start.'''
        self.resultText.config(text='Waiting for game')
        while True:
            if self.game_cancelled:
                self.cancel_game()
                self.find_game_button.config(state=ENABLED)
                self.cancel_stop_game_button.config(state=DISABLED, text=CANCEL_GAME_TEXT)
                break
            poll_results = self.poll_for_game_state()

            if poll_results['Result'] == 'SUCCESS':
                break
            if poll_results['Result'] == 'INVALID_PLAYER_KEY' or poll_results['Result'] == 'GAME_HAS_ENDED' or poll_results['Result'] == 'GAME_WAS_STOPPED':
                self.game_cancelled = True
            time.sleep(2)

    def play_game(self):
        ''' Play a game '''
        self.resultText.config(text='Playing game')
        self.in_game = True

        poll_results = self.poll_for_game_state()

        if poll_results['Result'] != 'SUCCESS': return

        game_state = poll_results['GameState']

        title = format('Game ID: ' + str(game_state['GameId']))
        game_style_details = self.game_styles_listbox.get('active').split(' | ')
        title += format(' / Style: ' + str(self.game_style_ids[int(self.game_styles_listbox.curselection()[0])]))
        title += format(' / Plates: ' + game_style_details[2].split(' ')[0])
        title += format(' / Guesses: ' + str(len(game_state['Nchoices'])))
        title += format(' / Timeout: ' + game_style_details[3])
        title += '\n'
        versus = format(self.bot_id + ' vs ' + game_state['OpponentId'])

        self.clear_game_title_text()
        self.set_game_title_text(title, '')
        self.set_game_title_text(versus, 'bold')
        self.middleFrame.update()

        while True:
            if self.game_cancelled: break

            if game_state['IsMover']:
                self.resultText.config(text='Playing Game - Your Turn')
                
                # @setStaticVars(processed=None, scores=None, images=None, plateData=None)
                move = npr_move.calculateMove(game_state)
                
                move_results = self.make_move(move)
                print(move_results['Result'])
                if move_results['Result'] == 'INVALID_MOVE':
                    self.resultText.config(text='Invalid Move')
                elif move_results['Result'] != 'SUCCESS':
                    self.resultText.config(text='Game has ended: ' + move_results['Result'])
                    print('Game ended')
                    break
                else:
                    game_state = move_results['GameState']
            else:
                self.resultText.config(text='Playing Game - Opponents Turn')

                # ---- Code here will be called on your opponent's turn ----
                # ----------------------------------------------------------

                poll_results = self.poll_for_game_state()

                if poll_results['Result'] != 'SUCCESS':
                    self.resultText.config(text='Game has ended: ' + poll_results['Result'])
                    print('Game ended')
                    break
                game_state = poll_results['GameState']

            if game_state['GameStatus'] != 'RUNNING': break
            self.middleFrameRight.update()

            delay = 0 #0.01
            try:
                te = int(self.thinking_time_entry.get())
                sleep = max((int(te)/1000, delay)) # Wait at least 10 ms
            except ValueError:
                sleep = delay
            time.sleep(sleep)

        self.set_in_game(False)

    def make_move(self, move):
        ''' Make a move '''
        req = {'BotId': self.bot_id, 'BotPassword': self.bot_password, 'PlayerKey': self.player_key, 'Move': move}
        url = BASE_URL + MAKE_MOVE_EXTENSION
        result = NPRDemoClient.make_api_call(url, req)
        # if result['Result'] == 'SUCCESS' or 'GAME_HAS_ENDED' in result['Result']:
        #     print(result)
        #     try:
        #         self.player.draw_game_state(result['GameState'], True)
        #         self.opponent.draw_game_state(result['GameState'], False)
        #     except Exception as e:
        #         print('Gamestate error: ' + str(e))
        return result

    def poll_for_game_state(self):
        ''' Poll the server for the latest GameState '''

        req = {'BotId': self.bot_id,
               'BotPassword': self.bot_password,
               'MaximumWaitTime': 1000,
               'PlayerKey': self.player_key}
        url = BASE_URL + POLL_FOR_GAME_STATE_EXTENSION

        result = NPRDemoClient.make_api_call(url, req)
        # if result['Result'] == 'SUCCESS' or 'GAME_HAS_ENDED' in result['Result']:
        #     self.player.draw_game_state(result['GameState'], True)
        #     self.opponent.draw_game_state(result['GameState'], False)
        return result

    def cancel_stop_game_clicked(self):
        self.game_cancelled = True
        self.cancel_game()
        self.find_game_button.config(state=ENABLED)
        self.cancel_stop_game_button.config(state=DISABLED, text=CANCEL_GAME_TEXT)

    def cancel_game(self):
        if self.player_key is None:
            return
        req = {'BotId': self.bot_id,
               'BotPassword': self.bot_password,
               'PlayerKey': self.player_key}

        url = BASE_URL + CANCEL_GAME_OFFER_EXTENSION
        NPRDemoClient.make_api_call(url, req)
        try:
            self.resultText.config(text='Cancelled game')
        except Exception as e:
            print(str(e) + ' -- resultText Message object no longer exists')

    @staticmethod
    def make_api_call(url, req):
        ''' Make an API call '''
        while True:
            try:
                res = requests.post(url, json=req, headers=API_CALL_HEADERS, timeout=60.0)
                try:
                    jres = res.json()
                    if 'Result' in jres:
                        return jres
                    time.sleep(0.1)
                except ValueError:
                    time.sleep(0.1)
            except requests.ConnectionError:
                time.sleep(0.1)
            except requests.Timeout:
                time.sleep(0.1)
            except requests.HTTPError:
                time.sleep(0.1)
            except BaseException as e:  # Bad code but needed for testing purposes
                print(e)
                time.sleep(0.1)


def int_with_commas(x):
    if type(x) not in [type(0), type(0)]:
        raise TypeError('Parameter must be an integer.')
    if x < 0:
        return '-' + int_with_commas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ',%03d%s' % (r, result)
    return '%d%s' % (x, result)


parser = argparse.ArgumentParser(description='Set optional running parameters')
parser.add_argument('--botid', default=None, help='log in with this bot name')
parser.add_argument('--password', default=None, help='log in with this password')
parser.add_argument('--autoplay', default=False, help='start playing automatically')
parser.add_argument('--gamestyle', default=None, help='play this gamestyle')
parser.add_argument('--timeout', default=0, help='have this timeout in milliseconds')
parser.add_argument('--playanothergame', action='store_true', help='Play another game when complete')
parser.add_argument('--dontplaysameuserbot', action='store_true', help='Don\'t play another user in the same account')
parser.add_argument('--closeaftergame', action='store_true', help='Close the client once the game has completed (takes priority over playanothergame)')
cmd_args = parser.parse_args()

# Hard code in a simple username/password for a time beeing
cmd_args.botid = '447xpro-defbot'
cmd_args.password = '447xpro-defbot'

root = tk.Tk()
my_gui = NPRDemoClient(root, cmd_args)
root.mainloop()
