#!/bin/sh

after=""
before=""
opts=""

while getopts ":A:B:o:" option; do
  case $option in
    A)
      after=${OPTARG}
      ;;
    B)
      before=${OPTARG}
      ;;
    o)
      opts=${OPTARG}
      ;;
  esac
done

for day in `ls var/scrap`
do
  if [[ "$day" > "$after" ]] && ([[ "$day" < "$before" ]] || [[ "$before" == "" ]])
  then
    python manage.py parse ${opts} ${day}
    sleep 1
  fi
done
