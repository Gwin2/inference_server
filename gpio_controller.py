import wiringpi as wpi
from wiringpi import GPIO
import os
import time
wpi.wiringPiSetup()

pins = [11, 4, 3, 14, 12, 0, 1, 2, 5, 8]
pin_jam = 7
length_to_jam = 10
time_to_hold_the_script = 4

on_state = 0
off_state = 1


gpio_flags_dir = "DRONE_DETECTION/NN_THRESHOLD/release/gpio_flags/"
flag_24 = 'ready_2.4'
flag_58 = 'ready_5.8'
length_24 = 'length_2.4'
length_58 = 'length_5.8'
jam = 'jam_indication'

check_jam = 0
def clear_jam_on_boot():
    with open(gpio_flags_dir + jam, 'w') as fl:
        fl.write('0')
    with open(gpio_flags_dir + length_24, 'w') as fl:
        fl.write('0')
    with open(gpio_flags_dir + length_58, 'w') as fl:
        fl.write('0')

for pin in pins:
    wpi.pinMode(pin, GPIO.OUTPUT)
    wpi.digitalWrite(pin, on_state)
wpi.pinMode(pin_jam, GPIO.OUTPUT)
wpi.digitalWrite(pin_jam, GPIO.LOW)
time.sleep(1)

for pin in pins:
    #wpi.pinMode(pin, GPIO.OUTPUT)
    wpi.digitalWrite(pin, off_state)


def indicate_distance_lightning(length) -> None:
    """
    Sets lightning according the ditance between the reciever and a drone.
    """
    global on_state
    global off_state

    if length > 0:
        for pin in pins[:length]:
            wpi.digitalWrite(pin, on_state)
        if length > length_to_jam:
            print('Включили глушилку!!!')
            wpi.digitalWrite(pin_jam, GPIO.HIGH)
            for pin in pins:
                wpi.digitalWrite(pin, off_state)
            wpi.digitalWrite(pins[len(pins)-2], on_state)
            wpi.digitalWrite(pins[len(pins)-1], on_state)
            with open(gpio_flags_dir + jam, 'w') as fl:
                fl.write(str(time_to_hold_the_script))
            
            time.sleep(time_to_hold_the_script)

            wpi.digitalWrite(pin_jam, GPIO.LOW)
            wpi.digitalWrite(pins[len(pins)-2], off_state)
            wpi.digitalWrite(pins[len(pins)-1], off_state)
            with open(gpio_flags_dir + jam, 'w') as fl:
                fl.write('0')
            with open(gpio_flags_dir + length_24, 'w') as fl:
                fl.write('0')
            with open(gpio_flags_dir + length_58, 'w') as fl:
                fl.write('0')
            print('Время вышло. Выключили глушилку!!!')

    if length != 10:
        for pin in pins[length:]:
            wpi.digitalWrite(pin, off_state)


while True:
    if check_jam == 0:
        clear_jam_on_boot()
        check_jam += 1

    files = os.listdir(gpio_flags_dir)
    while flag_24 not in files and flag_58 not in files:
        files = os.listdir(gpio_flags_dir)
    try:
        with open(gpio_flags_dir + length_24, 'r') as fl:
            l1 = fl.readline()
            l1 = l1.replace('\r', '').replace('\n', '')
            l1 = int(l1)
        
        with open(gpio_flags_dir + length_58, 'r') as fl:
            l2 = fl.readline()
            l2 = l2.replace('\r', '').replace('\n', '')
            l2 = int(l2)
        

        length = max(l1, l2)
        print('lengths: ', l1, l2)
        print(str(length), ' - length')
        indicate_distance_lightning(max(l1, l2))
    except Exception as e:
        print(str(e))
