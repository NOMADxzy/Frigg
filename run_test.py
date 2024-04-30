import yaml, subprocess
trace_list = ['ATT-LTE-driving', 'TMobile-LTE-driving', 'Verizon-EVDO-driving']
model_list = ['mfg', 'no_field', 'low_lstm_layer']
step_len_ms_list = [10,20,30]
sender_num_list = [1, 5, 20]

with open('template.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

command = [
                'python',
                './src/experiments/test.py',
                'local',
                '--schemes',
                'indigo_a3c_test',
                '--flows',
                '1'
            ]

for _, trace in enumerate(trace_list):
    for _,model in enumerate(model_list):
        for _, step_len_ms in enumerate(step_len_ms_list):
            for _, sender_num in enumerate(sender_num_list):
                config_data['trace'] = trace
                config_data['step_len_ms'] = step_len_ms
                config_data['flows'] = sender_num
                config_data['model_name'] = model
                with open('config.yaml', 'w') as file:
                    yaml.dump(config_data, file, default_flow_style=False)

                return_code = subprocess.call(command)