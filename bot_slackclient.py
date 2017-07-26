# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:12:02 2017

@author: voxxys
"""

from slackclient import SlackClient

slack_token = "xoxp-28316847441-28626652993-218947199479-1324bbaaeb5bddfa18fc6bee251ce2b4"
sc = SlackClient(slack_token)

def send_slack_message(message_text):
    sc.api_call("chat.postMessage", channel="#test", text = message_text)