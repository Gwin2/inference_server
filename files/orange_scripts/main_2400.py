from gnuradio import blocks
from gnuradio import gr
import sys
import signal
from files.orange_scripts import compose_send_data_2400 as my_freq
import osmosdr
import time
import threading
import subprocess
import os
import wiringpi as wpi
from wiringpi import GPIO

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path)

def light_diods_on_boot():
    pins = [11, 4, 3, 14, 12, 0, 1, 2, 5, 7]
    wpi.wiringPiSetup()

    for pin in pins:
        wpi.digitalWrite(pin, GPIO.HIGH)
        time.sleep(0.02)

    for pin in pins[::-1]:
        wpi.digitalWrite(pin, GPIO.LOW)
        time.sleep(0.02)


def get_hack_id():
    serial_number = os.getenv('hack')
    pos = None
    output = []
    try:
        command = 'lsusb -v -d 1d50:6089 | grep iSerial'
        output.append(subprocess.check_output(command, shell=True, text=True))
    except subprocess.CalledProcessError as e:
        print(f"Команда завершилась с кодом возврата {e.returncode}")
        print(e)
    print(output)
    output_lines = output[0].strip().split('\n')
    print(output_lines)
    serial_numbers = [line.split()[-1] for line in output_lines]
    print(serial_numbers)
    id = 0
    for i, number in enumerate(serial_numbers):
        if number == serial_number:
            id = i
            break
    if id is not None:
        print('HackId is: {0}'.format(id))
        return str(id)
    else:
        print('Такого хака нет!')

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
            args="numchan=" + str(1) + " " + 'hackrf=' + get_hack_id()
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
    time.sleep(3)
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
    tb.wait()


if __name__ == '__main__':
    main()