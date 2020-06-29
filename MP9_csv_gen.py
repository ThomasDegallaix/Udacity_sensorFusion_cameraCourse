import csv
import os

if __name__ == "__main__":

	curr_dir_path = os.path.dirname(os.path.realpath(__file__))
	target_dir_path = os.path.join(curr_dir_path, 'results/MP8_MP9_results/')

	listOfHeaders = ["Detector type", "Descriptor type", "Average matches", "Average time"]
	listOfValues = []

	for root,dirs,files in os.walk(target_dir_path):
		for file in files:

			#Temporary list which holds values for a row
			tmp_row = []

			split_result = file.split("_")
			detector_name = split_result[0]
			descriptor_name = split_result[1]
			tmp_row.append(detector_name)
			tmp_row.append(descriptor_name)


			with open(os.path.join(target_dir_path, file), mode='r') as f:
				reader = csv.DictReader(f)
				result = {}
				for row in reader:
					for column, value in row.items():
						result.setdefault(column, []).append(value)
				
				#Get matches average for each combination fo detector/descriptor
				matches_numbers = result.get("Matches number")
				sum_matches = 0
				for matches_number in matches_numbers:
					sum_matches += int(matches_number)
				avg_matches = sum_matches / len(matches_numbers)
				tmp_row.append(avg_matches)

				#Get time average for each combination fo detector/descriptor
			   	combination_times = result.get("Total time (ms)")
				sum_times = 0.0
				for time in combination_times:
					sum_times += float(time)
				avg_times = sum_times / len(combination_times)
				avg_times = format(avg_times, '.2f')
				tmp_row.append(avg_times)

			listOfValues.append(tmp_row)


	#Parse dict into a csv file
	with open(curr_dir_path+'/results/MP9_results.csv', 'w') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(listOfHeaders)
		writer.writerows(listOfValues)

