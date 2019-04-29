from sys import path
from os.path import dirname, realpath
MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)




import mysql.connector
import random
import csv

from sortedcontainers import SortedList


from datetime import datetime


class Score:
    def __init__(self, match_id, shooter_id, powerfactor=None, score=None, place=None, date=None):




        self.date = date
        self.place = place
        self.score = score
        self.match_id = match_id
        self.shooter_id = shooter_id
        self.powerfactor = powerfactor

    def __lt__(self, other):
        return self.date < other.date

    def __str__(self):
        return "{} {}".format(self.match_id, self.date)

    def __eq__(self, other):
        return self.match_id == other.match_id and self.shooter_id == other.shooter_id



def parse_date(date):
    for format in ('%Y-%m-%d', '%m-%d-%Y'):
        try:
            return datetime.strptime(date, format)
        except ValueError:
            pass
    raise ValueError("No valid date format found:", date)



def makeCSV(division, num_previous_matches = 5, data_size = 5000, noob_threshold = 100, min_match_size = 8):

    cnx = mysql.connector.connect(user='root', password='Atalanta',
                                  host='127.0.0.1',
                                  database='practiscore')
    cursor = cnx.cursor()

    print("Querying database. . . ")

    query = ("select * from practiscore.BB_" + division)
    cursor.execute(query)
    print("\tQuery complete!")


    shooters = dict()
    matches = dict()
    match_dates = dict()

    print("Populating datastructures. . . ")

    for line in cursor:
    # for i in range(3000):
        # line = cursor.fetchone()
        # print(line[-1])
        # print(line)

        match_id, shooter_id, powerfactor_string, score, place, date = line

        date = parse_date(date)

        # print(date)
        if powerfactor_string == "MAJOR":
            powerfactor = 1
        elif powerfactor_string == "MINOR":
            powerfactor = 0
        else:
            continue


        if not match_id in matches:
            matches[match_id] = dict()
            match_dates[match_id] = date
        matches[match_id][shooter_id] = ( Score(match_id, shooter_id, powerfactor, score, place, date) )

        if not shooter_id in shooters:
            shooters[shooter_id] = SortedList(key= lambda mid : match_dates[mid])
        shooters[shooter_id].add( match_id)

    cnx.close()

    # TODO if necessary, remove too-small matches from dictionary

    print("\tStructures complete!")



    # print(matches)
    # for shooter in shooters:
    #     print(shooter, ":")
    #     # print(shooter, " " , shooters[shooter][0])
    #     for match in shooters[shooter]:
    #         print("\t", match)

    filename = "/Data/" + division + "_" + str(num_previous_matches) + "_" + str(data_size) + ".csv"

    unique_hashes = set()

    with open(filename, 'wb') as csvfile:

        print("Writing to file {}:".format(filename))


        valid_entries = 0

        # for i in range(data_size):
        while valid_entries < data_size:
            # print(valid_entries)
            if valid_entries % round(data_size/100) == 0:
                print('.', end="", flush=True)

            rand_match_id = random.choice(list(matches.keys()))

            while len(matches[rand_match_id]) < min_match_size :
                rand_match_id = random.choice(list(matches.keys()))


            sh1, sh2 = random.sample( matches[rand_match_id].keys(), 2)

            label_score_1 = matches[rand_match_id][sh1]
            label_score_2 = matches[rand_match_id][sh2]


            index_of_split = shooters[sh1].index(rand_match_id)
            count = 0
            while (index_of_split < num_previous_matches or sh1 == sh2) and count < noob_threshold:
                sh1 = random.sample( matches[rand_match_id].keys(), 1)[0]
                label_score_1 = matches[rand_match_id][sh1]
                index_of_split = shooters[sh1].index(rand_match_id)
                count +=1
            if count >= noob_threshold or sh1 == sh2:
                continue

            sh1_history = shooters[sh1][:index_of_split]


            index_of_split = shooters[sh2].index(rand_match_id)
            count = 0
            while (index_of_split < num_previous_matches or sh1 == sh2) and count < noob_threshold:
                sh2 = random.sample( matches[rand_match_id].keys(), 1)[0]
                label_score_2 = matches[rand_match_id][sh2]
                index_of_split = shooters[sh2].index(rand_match_id)
                count += 1
            if count >= noob_threshold or sh1 == sh2:
                continue


            # Ensure Uniqueness----
            if sh1 < sh2:
                u_hash = str(rand_match_id) + " " +  str(sh1) + " "  + str(sh2)
            else:
                u_hash = str(rand_match_id) + " " + str(sh2) + " " + str(sh1)

            if u_hash in unique_hashes:
                continue
            else:
                unique_hashes.add(u_hash)

            # The combination is proven unique---

            valid_entries += 1
            sh2_history = shooters[sh2][:index_of_split]

            # print(sh1_history)
            # print(sh2_history)

            scorestring = ""
            for i in range(num_previous_matches):
                score = matches[sh1_history[i]][sh1]
                scorestring += str(score.score) + ","
                scorestring += str(len(matches[sh1_history[i]])) + ","
                scorestring += str(score.powerfactor) + ','

                score = matches[sh2_history[i]][sh2]
                scorestring += str(score.score) + ","
                scorestring += str(len(matches[sh2_history[i]])) + ","
                scorestring += str(score.powerfactor) + ','

            # scorestring += str(sh1) + "," + str(sh2) + ","
            scorestring += str(label_score_1.score / label_score_2.score) + ","
            scorestring += '0' if label_score_1.score <= label_score_2.score else '1'
            scorestring += "\n"
            # print(scorestring)
            csvfile.write(scorestring.encode('utf-8'))


        # print(valid_entries / data_size)
        print("\nFinished!\nTotal dataset: {}".format(valid_entries))


if __name__ == '__main__':

        num_previous_matches = 5
        data_size = 5000
        noob_threshold = 100
        min_match_size = 8
        # Little too small: 'revolver','classic',
        # Good: 'limited', 'all', 'single_stack', 'production', 'open', 'carry_optics'

        for division in ['limited', 'all', 'single_stack', 'production', 'open', 'carry_optics']:
            print("creating training file for {}".format(division))
            makeCSV(division, num_previous_matches, data_size, noob_threshold, min_match_size)