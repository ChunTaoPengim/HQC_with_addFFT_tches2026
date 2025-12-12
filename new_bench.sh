#!/bin/bash

# --- Determine if sudo is needed ---
if [ "$(uname)" = "Darwin" ]; then
    SUDO="sudo"
else
    SUDO=""
fi
# --- setup record file ---
if [ -z "$2" ]; then
  results_file="results.txt"
  echo "no target file name. default: $results_file"
else
  results_file="$2"
  echo "use result file: $results_file"
fi
echo "all results will be stored in : $results_file"


echo "========= HQC report  ==========" > "$results_file"
echo "current time: $(date)" >> "$results_file"
echo "================================" >> "$results_file"

uname -a | tee -a "$results_file"


# --- step 1: target setting ---
if [ -z "$1" ]; then
  target="ref"
  echo "no target. Use default : $target"
else
  target="$1"
  echo "set target: $target"
fi


# --- step 2: find and enter hqc-fft folder ---
if [ -d "hqc-fft" ]; then
  cd "hqc-fft" || { echo "error: cannot enter hqc-fft"; exit 1; }
  echo "entered folder: $(pwd)"
else
  echo "error: folder 'hqc-fft' not found in current directory."
  exit 1
fi


# --- HQC project list ---
hqc_projects=("hqc-1" "hqc-3" "hqc-5")


# --- start testing ---
for proj in "${hqc_projects[@]}"; do

  echo ""
  echo "--------------------------------------------------"
  echo ">>> processing project: $proj (TARGET=$target)"
  echo "--------------------------------------------------"

  echo -e "\n\n===== test starts: $proj =====\n" >> "../$results_file"

  # build project
  echo "--> make TARGET=$target PROJ=$proj ..."
  make TARGET="$target" PROJ="$proj" > /dev/null 2>&1

  # check make success
  if [ $? -ne 0 ]; then
      echo "error: 'make' fails for $proj"
      echo -e "\n===== 'make' fails in $proj =====\n" >> "../$results_file"
      continue
  fi

  # run gf2x-test
  echo "--> run ./bin/gf2x-test..."
  if [ -f "./bin/gf2x-test" ]; then
    echo -e "\n--- gf2x-test result ---\n" >> "../$results_file"
    $SUDO ./bin/gf2x-test >> "../$results_file" 2>&1
  else
    echo "error: can not find ./bin/gf2x-test"
    echo -e "\n--- error: can not find ./bin/gf2x-test ---\n" >> "../$results_file"
  fi

  # run code-test
  echo "--> run ./bin/code-test..."
  if [ -f "./bin/code-test" ]; then
    echo -e "\n--- code-test result ---\n" >> "../$results_file"
    $SUDO ./bin/code-test >> "../$results_file" 2>&1
  else
    echo "error: can not find ./bin/code-test"
    echo -e "\n--- error: can not find ./bin/code-test ---\n" >> "../$results_file"
  fi

  # run hqc-test
  echo "--> run ./bin/hqc-test..."
  if [ -f "./bin/hqc-test" ]; then
    echo -e "\n--- hqc-test result ---\n" >> "../$results_file"
    $SUDO ./bin/hqc-test >> "../$results_file" 2>&1
  else
    echo "error: can not find ./bin/hqc-test"
    echo -e "\n--- error: can not find ./bin/hqc-test ---\n" >> "../$results_file"
  fi

  make clean &> /dev/null
  echo ">>> finish $proj"

done


echo ""
echo "--------------------------------------------------"
echo "all test finished"
echo "check $results_file for result"
echo "--------------------------------------------------"

cat /proc/cpuinfo | tee -a "../$results_file"
