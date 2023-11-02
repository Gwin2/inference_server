#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: get_center_freq
# GNU Radio version: 3.8.1.0

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import my_freq  # embedded python module
import osmosdr
import time
import threading

import wiringpi as wpi
from wiringpi import GPIO


def light_diods_on_boot():

    #pins = [3, 4, 6, 9, 10, 7, 5, 2, 1, 0]
    pins = [11, 4, 3, 14, 12, 0, 1, 2, 5, 7]
    wpi.wiringPiSetup()

    for pin in pins:
        wpi.digitalWrite(pin, GPIO.HIGH)
        time.sleep(0.02)

    for pin in pins[::-1]:
        wpi.digitalWrite(pin, GPIO.LOW)
        time.sleep(0.02)


class get_center_freq(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "get_center_freq")

        ##################################################
        # Variables
        ##################################################
        self.prob_freq = prob_freq = 0
        self.top_peaks_amount = top_peaks_amount = 20
        self.samp_rate = samp_rate = 20e6
        self.poll_rate = poll_rate = 10000
        self.num_points = num_points = 8192
        self.flag = flag = 1
        self.decimation = decimation = 1
        self.center_freq = center_freq = my_freq.work(prob_freq)[0]

        ##################################################
        # Blocks
        ##################################################
        self.probSigVec = blocks.probe_signal_vc(4096)
        self.rtlsdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + 'hackrf=0'
        )
        self.rtlsdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.rtlsdr_source_0.set_sample_rate(samp_rate)
        self.rtlsdr_source_0.set_center_freq(center_freq, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_gain(100, 0)
        self.rtlsdr_source_0.set_if_gain(100, 0)
        self.rtlsdr_source_0.set_bb_gain(0, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
        self.rtlsdr_source_0.set_min_output_buffer(4096)
        def _prob_freq_probe():
            while True:

                val = self.probSigVec.level()
                try:
                    self.set_prob_freq(val)
                except AttributeError:
                    pass
                time.sleep(1.0 / (poll_rate))
        _prob_freq_thread = threading.Thread(target=_prob_freq_probe)
        _prob_freq_thread.daemon = True
        _prob_freq_thread.start()

        self.blocks_stream_to_vector_1 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 4096)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_stream_to_vector_1, 0), (self.probSigVec, 0))
        self.connect((self.rtlsdr_source_0, 0), (self.blocks_stream_to_vector_1, 0))

    def get_prob_freq(self):
        return self.prob_freq

    def set_prob_freq(self, prob_freq):
        self.prob_freq = prob_freq
        self.set_center_freq(my_freq.work(self.prob_freq)[0])

    def get_top_peaks_amount(self):
        return self.top_peaks_amount

    def set_top_peaks_amount(self, top_peaks_amount):
        self.top_peaks_amount = top_peaks_amount

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.rtlsdr_source_0.set_sample_rate(self.samp_rate)

    def get_poll_rate(self):
        return self.poll_rate

    def set_poll_rate(self, poll_rate):
        self.poll_rate = poll_rate

    def get_num_points(self):
        return self.num_points

    def set_num_points(self, num_points):
        self.num_points = num_points

    def get_flag(self):
        return self.flag

    def set_flag(self, flag):
        self.flag = flag

    def get_decimation(self):
        return self.decimation

    def set_decimation(self, decimation):
        self.decimation = decimation

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.rtlsdr_source_0.set_center_freq(self.center_freq, 0)



def main(top_block_cls=get_center_freq, options=None):
    #for k in range(0, 3):
    #    light_diods_on_boot()

    tb = top_block_cls()
    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    #tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
