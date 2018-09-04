def split_data(data, ratio):

	len_data = len(data['labels'])
	assert len_data == len(data['labels'])

	len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)
	print(len_train, len_valid, len_test)

	splited_data = {'train': dict(), 'valid': dict(), 'test': dict()}
	
	for key in data:
		splited_data['train'][key] = data[key][ : len_train]
		splited_data['valid'][key] = data[key][len_train : len_train + len_valid]
		splited_data['test'][key] = data[key][len_train + len_valid : ]

	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])

	return splited_data