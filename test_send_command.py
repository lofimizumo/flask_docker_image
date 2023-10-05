from util import PriceAndLoadMonitor

controller = PriceAndLoadMonitor(sn='RX2505ACA10JOA160037')
controller.send_battery_command('Discharge')