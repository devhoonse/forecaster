#!/bin/bash


echo "cat /dev/null > /var/spool/mail/nfa2020"
cat /dev/null > /var/spool/mail/nfa2020

echo "cat /dev/null > $1/nohup.out"
cat /dev/null > $1/nohup.out
