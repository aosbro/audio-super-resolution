# Ask user's confirmation
read -p "Create new conda env ([Y]/n)?" continue

if [ "$continue" == "n" ]; then
  echo "exit";
else
  # Ask user conda environment's name
  echo "Creating new conda environment, choose name"
  read env_name
  echo "Name $env_name was chosen";

  # Create and activate the conda environment with proper name
  conda create --name $env_name python=3.6
  conda activate $env_name

  # Install the project's requirement
  pip install -r requirements.txt
fi