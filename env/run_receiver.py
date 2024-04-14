#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan, Jestin Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import argparse
import time

from receiver import Receiver
import threading

def run_receiver(receiver):
    receiver.handshake()
    receiver.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()


    threads = []
    receivers = []
    for i in range(0,3):
        receiver = Receiver(args.ip, args.port + i)
        threads.append(threading.Thread(target=run_receiver, args=(receiver,)))
        receivers.append(receiver)

    try:
        for thread in threads:
            thread.start()

        time.sleep(100)
    except KeyboardInterrupt:
        pass
    finally:
        for receiver in receivers:
            receiver.cleanup()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('ip', metavar='IP')
#     parser.add_argument('port', type=int)
#     args = parser.parse_args()
#
#     receiver = Receiver(args.ip, args.port)
#
#     try:
#         receiver.handshake()
#         receiver.run()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         receiver.cleanup()

if __name__ == '__main__':
    main()
