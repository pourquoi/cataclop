#!/bin/bash

Help () {
  echo "Load or create scrap archives."
  echo ""
  echo "archive.sh action [-k] [-d]"
  echo "  action:"
  echo "    create    archive the scrapped races"
  echo "    load      load the archived races"
  echo "  options:"
  echo "    -k    do not delete source files after archive creation"
  echo "    -d    date or date prefix"
  echo ""
  echo "  examples:"
  echo "  archive all scrapped races and delete them:"
  echo "    archive.sh create"
  echo "  archive a given month:"
  echo "    archive.sh create -d \"2021-01\""
  echo "  archive a given month and keep the source files:"
  echo "    archive.sh create -k -d \"2021-01\""
  echo "  load all archives:"
  echo "    archive.sh load"
  echo "  load one month:"
  echo "    archive.sh load -d \"2021-01\""
}

Create () {
  if [ "$dateprefix" == "" ]
  then
    from=$(ls var/scrap/ | head -n 1)
    to=$(ls var/scrap/ | tail -n 1)
    tar -czvf var/archives/${from}-${to}.tar.gz var/scrap/
  else
    tar -czvf var/archives/${dateprefix}.tar.gz var/scrap/${dateprefix}*
  fi

  if [ $keep -eq 0 ]
  then
    LASTSCRAP=$(ls var/scrap/ | tail -n 1)
    if [ $LASTSCRAP == "" ] ; then return ; fi
    mv var/scrap/${LASTSCRAP} var/last_scrap

    if [ "$dateprefix" == "" ]
    then
      from=$(ls var/scrap/ | head -n 1)
      to=$(ls var/scrap/ | tail -n 1)
      rm -rf var/scrap/*
    else
      rm -rf var/scrap/${dateprefix}*
    fi

    mv var/last_scrap var/scrap/${LASTSCRAP}
  fi
}

Load () {
  if [ "$dateprefix" == "" ]
  then
    racedirs=`ls var/archives/*.tar.gz`
  else
    racedirs=`ls var/archives/${dateprefix}.tar.gz`
  fi

  for archive in $racedirs
  do
    tar -xzvf "$archive"
  done
}

if [ ! -d "var/archives" ] 
then
  echo "archives directory not found"
  echo ""
  Help
  return
fi

keep=0
dateprefix=""
action=$1
shift

while getopts ":hkd:" option; do
  case $option in
    k)
      keep=1
      ;;
    d)
      dateprefix=${OPTARG}
      ;;
    h)
      Help
      exit;;
  esac
done
shift $((OPTIND -1))

case $action in
  "create")
    Create
    ;;
  "load")
    Load
    ;;
  *)
    echo "action missing"
    echo ""
    Help
    exit;;
esac
