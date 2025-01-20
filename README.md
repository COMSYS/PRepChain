# PRepChain: A Versatile Privacy-Preserving Reputation System for Dynamic Supply Chain Environments

## About
This is a Flask-based prototype implementing our reputation system "PRepChain". Each entity is represented by a Flask instance running on different ports on the same machine. Two scripts concerning the rating and query process utilize the entity interface to automatize the respective voter-/inquirer-initiated process.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

If you are planning to integrate parts of our work into a commercial product and do not want to disclose your source code, please contact us for other licensing options via email at pennekamp (at) comsys (dot) rwth-aachen (dot) de

## Acknowledgments

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy – EXC-2023 Internet of Production – 390621612.

# Installation

## Prerequisites

1. **Supported OS**:
   - Windows 11 Pro (Version 24H2)
   - Ubuntu Server

2. **Python Installation**:
   - Ensure that Python is installed on your system. The recommended version is Python 3.8 or later.
   - You can download Python from the [official website](https://www.python.org/).
   - Verify the installation by running:
     ```bash
     python --version
     ```

3. **MongoDB Installation**:
   - Install MongoDB Community Server from the [official MongoDB website](https://www.mongodb.com/try/download/community).
   - Follow the installation instructions specific to your operating system.
   - After installation, ensure MongoDB is running locally on the default port `27017`.
   - Start the MongoDB service (if not started automatically):
     ```bash
     mongod
     ```
   - Verify the connection using:
     ```bash
     mongo
     ```

## Installing Project Dependencies

1. **Clone the Repository**:
   - Use the following command to clone the repository:
     ```bash
     git clone https://github.com/COMSYS/PRepChain.git
     ```

2. **Create a Virtual Environment** (optional but recommended):
   - Create a virtual environment:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

3. **Install Dependencies**:
   - Install the required packages using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

4. **Verify the Installation**:
   - Ensure all packages are installed correctly by running:
     ```bash
     pip list
     ```

You are now ready to run the project. If you encounter any issues, consult the documentation or raise an issue in the repository.

# Usage
- start_entities.bat is a script that automatically starts all required entity Flask instances (Windows).
- start_entities.sh is a script that automatically starts all required entity Flask instances (Linux).
- reputationsystem/reset_all.py initializes the databases and folder structure and also resets these components to the default state. 

**To run the evaluation scripts, first you have to edit the config.toml accordingly:**

```
[DEBUG]
flask_debug = false #This should be set to false otherwise the verification engine instance crashes after several runs

[Rating]
rating_fields = ["sub1", "sub2", "sub3", "goodsreceipt", "temperatursaegeblatt", "schwingung", "vortrieb", "trustedgoodsreceipt", "trustedtemperatursaegeblatt", "trustedschwingung", "trustedvortrieb"] #this must be a combination of sub_fields, obj_fields, tru_fields
sub_fields = ["sub1", "sub2", "sub3"] #the subjective fields in the rating
obj_fields = ["goodsreceipt", "temperatursaegeblatt", "schwingung", "vortrieb"] #the objective (untrusted) fields in the rating, corresponding to the files in the votee folder
tru_fields = ["trustedgoodsreceipt", "trustedtemperatursaegeblatt", "trustedschwingung", "trustedvortrieb"] #the objective (trusted) fields in the rating, corresponding to the files in the votee folder
rating_weights = [0.5,0.8,1]
eq_classes_num = 3
eq_classes = [[0,3.33],[3.33,6.66],[6.66,10]]
sub_num = 3 #must be the length of sub_fields
sub_upperbound = 10
sub_lowerbound = 1
obj_num = 4 #must be the length of obj_fields
tru_num = 4 #must be the length of tru_fields
```

- **The evaluation data of each entity and the rating or query process is stored in the folder "evaluation/eval_data/eval_x" where x corresponds to the respective evaluation script.**
- **Each evaluation run is executed 20 times.**

### ratingprocess_eval1.py
- Here we test the number of max aggregations possible. To this end, we utilize an infinite loop where a singular voter rates the same votee repeatedly.

### ratingprocess_eval2.py
- Here, we test the influence of the amount of subjective, objective and objective_trusted rating fields on the rating process duration while keeping the system-wide specified amount of available rating fields fixed. To this end, we differentiate between the four voter/votee reputation cases.

### ratingprocess_eval3.py
- Here, we test the influence of the amount of subjective rating fields on the rating process duration while the system-wide specified subjective rating fields equals the amount of the respective evaluation run. To this end, we differentiate between the four voter/votee reputation cases.
- Note, that the config.toml must be modified for each run in the following way:

```
[Rating]
rating_fields = ["sub1"] #first run
rating_fields = ["sub1","sub2"] #second run (delete the first run rating_fields)
sub_fields = ["sub1"] #first run
sub_fields = ["sub1", "sub2"] #second run (delete the first run sub_fields)
obj_fields = []
tru_fields = []
rating_weights = [0.5,0.8,1]
eq_classes_num = 3
eq_classes = [[0,3.33],[3.33,6.66],[6.66,10]]
sub_num = 1 #first run
sub_num = 2 #second run (delete the first run sub_num)
sub_upperbound = 10
sub_lowerbound = 1
obj_num = 0
tru_num = 0
```

- Also the ratingprocess_eval3.py file needs to be modified in the following way before each run:

```
if os.path.exists(data_file_path):
        with open(data_file_path, 'rb') as f:
            data = pickle.load(f)
            current_start = 20001 #Eval run x dann current_start = 20"x"01
            current_end = 20100 #Eval run x dann current_end = 20"x+1"00
            users = {i: 0 for i in range(current_start, current_end + 1)}
            iteration = data['iteration']
            
```

### ratingprocess_eval4.py
- This rating process script serves as basis for testing different subjective rating fields amounts in addition to specific objective data files with different files sizes. To this end, we differentiate between the four voter/votee reputation cases.
- Note, that the config file needs to be modified accordingly for each evaluation case.

### queryprocess.py
- Here we test the query process. 
- Edit the variable evaluation_name with the corresponding evaluation run in the rating process to make sure that the performance measurements are stored in the correct evaluation folder. 
