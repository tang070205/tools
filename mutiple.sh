#!/bin/bash
### HOW TO USE #################################################################################
### SYNTAX: ./outcar2nep-exyz.sh dire_name
###     NOTE: 1).'dire_name' is the directory containing OUTCARs
### Email: yanzhowang@gmail.com if any questions
### Modified by Yuwen Zhang
### Modified by Shunda Chen
### Modified by Zihan Yan
################################################################################################
#--- DEFAULT ASSIGNMENTS ---------------------------------------------------------------------
isol_ener=0     # Shifted energy, specify the value?
viri_logi=1     # Logical value for virial, true=1, false=0
#--------------------------------------------------------------------------------------------
read_dire=$1
if [ -z "$read_dire" ]; then
        echo "Your syntax is illegal, please try again"
        exit 1
fi

writ_dire="NEPdataset"; writ_file="NEP-dataset.xyz";
rm -rf $writ_dire; mkdir $writ_dire


for i in `find -L $read_dire -name "MD_dump"`
do
    configuration=$(echo "$i" |sed 's/\/MD_dump//g' | awk -F'/' '{print $(NF-2)"/"$(NF-1)"/"$NF}')
    syst_numb_atom=$(tail -n 3 $i | head -n 1 | awk '{print $1+1}')
    mdstep_lines=(grep -n "MDSTEP" $i | awk -F: '{print $1+13+'$syst_numb_atom'}')
    ener=$(grep 'final etot' running_md.log | awk '{print $4}')
    log_dir=$(echo "$i" | sed 's/\/MD_dump//g')
    log_file="${log_dir}/running_md.log"
    conversion_value=$(grep "Volume (A^3)" "$log_file" | awk '{print $4/1602.1766208}')

    for ((i=0; i<${#mdstep_lines[@]}; i++)); do
        start_line=${mdstep_lines[i]}
        end_line=${mdstep_lines[i+1]}

        sed -n "${start_line},${end_line}p" "$file" > temp.file
        echo "$syst_numb_atom" >> "$writ_dire/$writ_file"
        latt=$(grep -A 4 "LATTICE_VECTORS" temp.file | tail -n 3 | sed 's/-/ -/g' | awk '{print $1,$2,$3}' | xargs)

        if [[ $viri_logi -eq 1 ]]; then
            viri=$(grep -A 4 "VIRIAL (kbar)" temp.file | tail -n 3 | awk '{for (i = 1; i <= NF; i++) {printf  $i * '$conversion_value'}}' |xargs)
            echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Energy=${ener[i+1]} Virial=\"$viri\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
        else
            echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Properties=species:S:1:pos:R:3:forces:R:3 Energy=${ener[i+1]} pbc=\"T T T\"" >> "$writ_dire/$writ_file"
        fi

        grep -A $((syst_numb_atom + 1)) "INDEX" temp.file | tail -n $syst_numb_atom > "$writ_dire/$writ_file"
    done
    rm -f temp.file

done

dos2unix "$writ_dire/$writ_file"
echo "All done."
