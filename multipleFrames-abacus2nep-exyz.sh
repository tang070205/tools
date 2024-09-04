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

writ_dire="NEPdataset-multiple_frames"
writ_file="NEP-dataset.xyz"
error_file="non_converged_files.txt"

rm -rf "$writ_dire"
mkdir "$writ_dire"
rm -f "$error_file"

N_count=1
for file in "${converged_files[@]}"; do
    configuration=$(basename "$(dirname "$file")")
    start_lines=($(sed -n '/aborting loop because EDIFF is reached/=' "$file"))
    end_lines=($(sed -n '/[^ML] energy  without entropy/=' "$file"))
    ion_numb_arra=($(grep "ions per type" "$file" | tail -n 1 | awk -F"=" '{print $2}'))
    ion_symb_arra=($(grep "POTCAR:" "$file" | awk '{print $3}' | awk -F"_" '{print $1}' | awk '!seen[$0]++'))
    syst_numb_atom=$(grep "number of ions" "$file" | awk '{print $12}')

    for ((i=0; i<${#mdstep_lines[@]}; i++)); do
        start_line=${mdstep_lines[i]}
        end_line=${mdstep_lines[i+1]}

        sed -n "${start_line},${end_line}p" "$file" > temp.file
        echo "$syst_numb_atom" >> "$writ_dire/$writ_file"
        latt=$(grep -A 7 "VOLUME and BASIS-vectors are now" temp.file | tail -n 3 | sed 's/-/ -/g' | awk '{print $1,$2,$3}' | xargs)
        ener=$(grep "free  energy   TOTEN" temp.file | tail -1 | awk '{printf "%.10f\n", $5 - '"$syst_numb_atom"' * '"$isol_ener"'}')

        if [[ $viri_logi -eq 1 ]]; then
            viri=$(grep -A 20 "FORCE on cell =-STRESS" temp.file | grep "Total " | tail -n 1 | awk '{print $2,$5,$7,$5,$3,$6,$7,$6,$4}')
            echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Virial=\"$viri\" pbc=\"T T T\" Properties=species:S:1:pos:R:3:forces:R:3" >> "$writ_dire/$writ_file"
        else
            echo "Config_type=$configuration Weight=1.0 Lattice=\"$latt\" Properties=species:S:1:pos:R:3:forces:R:3 pbc=\"T T T\"" >> "$writ_dire/$writ_file"
        fi

        for((j=0;j<${#ion_numb_arra[*]};j++));do
            printf ''${ion_symb_arra[j]}'%.0s\n' $(seq 1 1 ${ion_numb_arra[j]}) >> $writ_dire/symb.tem
        done

        grep -A $((syst_numb_atom + 1)) "TOTAL-FORCE (eV/Angst)" temp.file | tail -n $syst_numb_atom > "$writ_dire/posi_forc.tem"
        paste "$writ_dire/symb.tem" "$writ_dire/posi_forc.tem" >> "$writ_dire/$writ_file"
        rm -f "$writ_dire"/*.tem
    done
    rm -f temp.file

done

echo -ne "\nConversion complete.\n"
dos2unix "$writ_dire/$writ_file"
echo "All done."
