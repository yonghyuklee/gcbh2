from ase.optimize.optimize import Dynamics
from ase import units
from pygcga2.utilities import NoReasonableStructureFound
from ase.io import read
from ase.io import write
import json
from time import strftime, localtime
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
import subprocess
import os
import sys
import shutil
# from distutils.version import LooseVersion
import pickle as pckl
from ase.db import connect
from ase.neighborlist import NeighborList, natural_cutoffs #, get_connectivity_matrix
from ase.data import covalent_radii as covalent
from ase.data import atomic_numbers, atomic_masses
# from quippy.potential import Potential
from mpi4py import MPI
from lammps import PyLammps
# from ase.calculators.lammpslib import LAMMPSlib
# from ase.build import molecule
# from ase.io.trajectory import TrajectoryWriter

import numpy as np
from scipy import sparse

# print(np.version.version)
# assert LooseVersion(np.version.version) > LooseVersion("1.7.0")


class UnreasonableStructureFound(Exception):
    pass


class FragmentedStructure(Exception):
    pass


programlogo = r"""
                                               ,--, 
    ,----..      ,----..       ,---,.        ,--.'| 
   /   /   \    /   /   \    ,'  .'  \    ,--,  | : 
  |   :     :  |   :     : ,---.' .' | ,---.'|  : ' 
  .   |  ;. /  .   |  ;. / |   |  |: | |   | : _' | 
  .   ; /--`   .   ; /--`  :   :  :  / :   : |.'  | 
  ;   | ;  __  ;   | ;     :   |    ;  |   ' '  ; : 
  |   : |.' .' |   : |     |   :     \ '   |  .'. | 
  .   | '_.' : .   | '___  |   |   . | |   | :  | ' 
  '   ; : \  | '   ; : .'| '   :  '; | '   : |  : ; 
  '   | '/  .' '   | '/  : |   |  | ;  |   | '  ,/  
  |   :    /   |   :    /  |   :   /   ;   : ;--'   
   \   \ .'     \   \ .'   |   | ,'    |   ,/       
    `---`        `---`     `----'      '---'        


          Grand Canonical Basin-Hoppings 
                 Geng Sun(UCLA)          
                gengsun@ucla.edu        
---------------------------------------------------
"""


def get_current_time():
    time_label = strftime("%d-%b-%Y %H:%M:%S", localtime())
    return time_label


class GrandCanonicalBasinHopping(Dynamics):
    """Basin hopping algorithm.

    After Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116

    and

    David J. Wales and Harold A. Scheraga, Science, Vol. 285, 1368 (1999)
    """

    def __init__(
        self,
        atoms,
        temperature=1500.0,
        t_nve=1500,
        maximum_temp=None,
        minimum_temp=None,
        stop_steps=50,
        logfile="grandcanonical.log",
        trajectory="grandcanonical.db",
        local_minima_trajectory="local_minima.traj",
        local_minima_trajecotry_db="local_minima.db",
        adjust_cm=False,
        restart=False,
        chemical_potential=None,
        bash_script="optimize.sh",
        model_file=None,
        model_label=None,
        files_to_copied=None,
        elements=None,
    ):
        """Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        """

        self.t_nve = t_nve
        self.T = temperature
        if maximum_temp is None:
            self.max_T = 1.0 / ((1.0 / self.T) / 1.5)
        else:
            self.max_T = max([maximum_temp, self.T])
        if minimum_temp is None:
            self.min_T = 1.0 / ((1.0 / self.T) * 1.5)
        else:
            self.min_T = min([minimum_temp, self.T])
        self.stop_steps = stop_steps
        self.restart = restart
        self.bash_script = bash_script
        self.copied_files = files_to_copied

        self.model_file = model_file
        self.model_label = model_label
        self.elements = elements
        self.cmds = None

        # some file names and folders are hardcoded
        self.fn_current_atoms = "Current_atoms.traj"
        self.fn_status_file = "Current_Status.json"
        self.opt_folder = "opt_folder"

        self.structure_modifiers = {}

        self.adjust_cm = adjust_cm

        if 0:
            self.lm_trajectory = local_minima_trajectory
            if isinstance(local_minima_trajectory, str):
                self.lm_trajectory = Trajectory(local_minima_trajectory, "a", atoms)
        self.lm_trajectory = connect(local_minima_trajecotry_db)
        self.opt_trajectory = connect(trajectory)

        # Dynamics.__init__ simply set
        # self.atoms and
        # self.logfile and
        # self.trajectory and
        # self.nsteps
        Dynamics.__init__(self, atoms, logfile, trajectory)

        # print the program logo at the beginning of the output file
        self.logfile.write("%s\n" % programlogo)
        self.logfile.flush()

        # setup the chemical potential for different elements
        self.mu = {}
        if chemical_potential is not None and os.path.isfile(chemical_potential):
            with open(chemical_potential, "r") as fp:
                for i, istr in enumerate(fp):
                    if istr.strip() == "":
                        continue
                    if istr.startswith("#"):
                        continue
                    k, v = istr.split()
                    self.mu[k] = float(v)
        else:
            raise RuntimeError(
                "chemical potential file %s is not found" % chemical_potential
            )
        for k, v in self.mu.items():
            self.dumplog("Chemical potential of %s is %.3f" % (k, v))

        # try to read previous result
        if self.restart:
            if (not os.path.isfile(self.fn_status_file)) or (
                not os.path.isfile(self.fn_current_atoms)
            ):
                self.dumplog(
                    "%s or %s no found, start from scratch\n"
                    % (self.fn_current_atoms, self.fn_status_file)
                )
                self.restart = False
            elif os.path.getsize(self.fn_current_atoms) == 0:
                self.dumplog(
                    "{} is empty, set self.restart=False".format(self.fn_current_atoms)
                )
                self.restart = False
            else:
                try:
                    atoms = read(self.fn_current_atoms)
                    atoms.get_potential_energy()
                except PropertyNotImplementedError:
                    self.dumplog(
                        "No energy found in {}, set self.restart=False".format(
                            self.fn_current_atoms
                        )
                    )
                    self.restart = False
                except RuntimeError as e:
                    self.dumplog("Error when read {}, set self.restart=False".format(e))
                    self.restart = False

        self.energy = None
        self.free_energy = None
        self.energy_min = None
        self.free_energy_min = None
        self.no_improvement_step = 0
        # negative value indicates no on-going structure optimization, otherwise it will be the  on-going optimization
        self.on_optimization = -1

        # this is used for adjusting the temperature of Metropolis algorithm
        self.accept_history = (
            []
        )  # a series of 0 and 1, 0 stands for not accpeted, 1 stands for accepted
        self.accept_history_full = (
            []
        )  # a series of 0 and 1, 0 stands for not accpeted, 1 stands for accepted
        self.max_history = 25  # max length of self.accept_history is 25

        if not self.restart:
            self.initialize()
        else:
            self.reload_previous_results()

    def todict(self):
        d = {}
        return d

    def dumplog(self, msg="", level=1, highlight=None):
        if level < 1:
            level = 1
        real_message = " " * level + msg.strip() + "\n"
        if highlight is None:
            self.logfile.write(real_message)
        else:
            bars = highlight * (len(real_message) - 1) + "\n"
            self.logfile.write(bars)
            self.logfile.write(real_message)
            self.logfile.write(bars)
        self.logfile.flush()

    def initialize(self):
        self.on_optimization = 0
        self.nsteps = 0
        self.rejected_steps = 0
        self.optimize(self.atoms)
        self.save_current_status()
        self.energy = self.atoms.get_potential_energy()
        ref = self.get_ref_potential(self.atoms)
        self.free_energy = self.energy - ref
        self.energy_min = self.energy
        self.free_energy_min = self.free_energy
        self.no_improvement_step = 0
        self.on_optimization = -1
        self.save_current_status()
        self.nsteps += 1
        self.modifier_name = ""

    def save_current_status(self):
        # save current atoms
        t = self.atoms.copy()
        t.info = self.atoms.info.copy()
        e = self.atoms.get_potential_energy()
        f = self.atoms.get_forces()
        spc = SinglePointCalculator(t, energy=e, forces=f)
        t.set_calculator(spc)
        write(self.fn_current_atoms, t)

        accept_digits = ""
        for ii in self.accept_history:
            accept_digits += str(ii)
            accept_digits += ","
        accept_digits = accept_digits[:-1]

        if self.nsteps > 1:
            if self.accept_history[-1] == 0 and self.modifier_name == "nve":
                self.t_nve = self.t_nve * 1.005
            # if self.accept_history[-1]==1:
            #     self.t_nve= self.t_nve/1.005

        # save the current status of the basin hopping
        if self.nsteps == 0:
            info = {
                "nsteps": self.nsteps,
                "no_improvement_step": self.no_improvement_step,
                "Temperature": self.T,
                "free_energy_min": self.free_energy_min,
                "energy_min": self.energy_min,
                "history": accept_digits,
                "on_optimization": self.on_optimization,
                "t_nve": self.t_nve,
                "modifier_name": "",
                "rejection_rate": 0,
            }
        else:
            info = {
                "nsteps": self.nsteps,
                "no_improvement_step": self.no_improvement_step,
                "Temperature": self.T,
                "free_energy_min": self.free_energy_min,
                "energy_min": self.energy_min,
                "history": accept_digits,
                "on_optimization": self.on_optimization,
                "t_nve": self.t_nve,
                "modifier_name": self.modifier_name,
                "rejection_rate": self.rejected_steps / self.nsteps,
            }

        with open(self.fn_status_file, "w") as fp:
            json.dump(info, fp, sort_keys=True, indent=4, separators=(",", ": "))

    def reload_previous_results(self):
        with open(self.fn_status_file) as fp:
            info = json.load(fp)
            for k, v in info.items():
                if hasattr(v, "keys"):
                    # if v is also a dictionary, which is used for recording the weights of operators; but they are not
                    # saved in the current version
                    self.dumplog("Read in {}".format(k))
                    for sub_k, sub_v in v.items():
                        self.dumplog("{0}={1}".format(sub_k, sub_v), level=4)
                else:
                    self.dumplog("Read previous result {0} ={1}".format(k, v))
            tl = get_current_time()
            self.dumplog("### %s: Previous Status Read in Successfullly ###\n" % tl)
            self.nsteps = info["nsteps"]
            self.no_improvement_step = info["no_improvement_step"]
            self.free_energy_min = info["free_energy_min"]
            self.energy_min = info["energy_min"]
            self.rejected_steps = int(info["rejection_rate"] * self.nsteps)

            if self.nsteps > 0:
                self.modifier_name = info["modifier_name"]
            # Temperature and history is collected
            # since some previous version does not have this two terms, we have to query about the existence.
            if "Temperature" in info.keys():
                self.dumplog("Previous temperature is read\n")
                self.T = info["Temperature"]
            if "history" in info.keys():
                for ii in info["history"].split(","):
                    if ii.isdigit():
                        self.accept_history.append(int(ii))

            if "full_history" in info.keys():
                for ii in info["full_history"].split(","):
                    if ii.isdigit():
                        self.full_history.append(int(ii))

            if "on_optimization" in info.keys():
                self.on_optimization = info["on_optimization"]

        previous_atoms = read(self.fn_current_atoms)
        self.update_self_atoms(previous_atoms)
        # get the self.energy and self.free_energy
        self.energy = self.atoms.get_potential_energy()
        ref = self.get_ref_potential(self.atoms)
        self.free_energy = self.energy - ref
        self.dumplog("self.atoms read successfully")

        # try to relocate previous optimization result
        if self.on_optimization > -1:
            opt_folder = os.path.join(
                os.getcwd(), self.opt_folder, "opt_%05d" % self.on_optimization
            )
            assert os.path.isdir(opt_folder)
            self.nsteps = self.on_optimization
            a = previous_atoms.copy()
            self.save_current_status()
            self.optimize(inatoms=a)
            self.accepting_new_structures(newatoms=a)
            self.on_optimization = -1
            self.save_current_status()
            self.nsteps += 1
        else:
            self.dumplog("Start new optimization from current atoms")

    def add_modifier(self, func, name="mutation", weight=1.0, *args, **kwargs):
        if not hasattr(func, "__call__"):
            raise RuntimeError("modifier must be a function")
        elif name in self.structure_modifiers.keys():
            raise RuntimeError("structure modifier %s exists already!\n" % name)
        self.structure_modifiers[name] = [
            func,
            args,
            kwargs,
            weight,
            weight,
        ]  # second weight is variable

    def select_modifier(self):
        operator_names = self.structure_modifiers.keys()
        if not isinstance(operator_names, list):
            operator_names = list(operator_names)
        operator_weights = np.asarray(
            [self.structure_modifiers[key][-1] for key in operator_names]
        )
        # operator_weights = operator_weights/operator_weights.sum()
        # return np.random.choice(operator_names, p=operator_weights)
        # sum_of_weights = sum(operator_weights)
        cum_sum_weights = np.cumsum(operator_weights)
        p = np.random.uniform(low=0.0, high=operator_weights.sum())
        for index in range(0, len(operator_names)):
            if p < cum_sum_weights[index]:
                return operator_names[index]
        return operator_names[-1]

    def update_modifier_weights(self, name="mutation", action="increase"):
        if name not in self.structure_modifiers.keys():
            raise RuntimeError("operator name %s not recognized" % name)
        if action not in ["increase", "decrease", "reset"]:
            raise RuntimeError("action must be 'increase','decrease' or 'rest'")
        elif action == "reset":
            for key, values in self.structure_modifiers.items():
                values[-1] = values[-2]
            self.dumplog("All the modifier weights are reset as 1.0\n")
        elif action == "increase":
            w = self.structure_modifiers[name][-1]
            w_orginal = self.structure_modifiers[name][-2]
            self.structure_modifiers[name][-1] = min([w_orginal * 2.0, w * 1.05])
        else:
            w = self.structure_modifiers[name][-1]
            w_orginal = self.structure_modifiers[name][-2]
            self.structure_modifiers[name][-1] = max([w_orginal / 2.0, w / 1.05])

    def move(self, modifier_name="mutation"):
        """Move atoms by a random step."""
        atoms = self.atoms.copy()
        self.dumplog(
            "%s : Starting operator '%s' (formula %s) \n"
            % (get_current_time(), modifier_name, atoms.get_chemical_formula())
        )
        func, arg, kwargs, weight_orginal, weight = self.structure_modifiers[
            modifier_name
        ]
        atoms = func(
            atoms, *arg, **kwargs
        )  # to be careful, func always accepts atoms as the first argument.
        if self.adjust_cm:
            atoms.center()
        self.dumplog(
            "%s : End operator (formula %s) \n"
            % (get_current_time(), atoms.get_chemical_formula())
        )
        return atoms

    def log_status(self):
        time_label = get_current_time()
        natoms = self.atoms.get_global_number_of_atoms()
        formula = self.atoms.get_chemical_formula()
        self.dumplog(
            "%20s%6s (natoms=%3d, %8s) Steps:%8d E=%15.8f F=%15.8f \n"
            % (
                time_label,
                "GCBH",
                natoms,
                formula,
                self.nsteps - 1,
                self.energy,
                self.free_energy,
            )
        )
        for key in self.structure_modifiers.keys():
            self.dumplog(
                "modifier %s (weight %3.2f)    "
                % (key, self.structure_modifiers[key][-1])
            )
        self.dumplog("Current Temperature is %.2f" % self.T)

    def run(self, maximum_steps=4000, maximum_trial=50, multiple=False, n_multiple=20):
        """Hop the basins for defined number of steps."""
        if multiple:
            while self.nsteps < maximum_steps:
                if self.no_improvement_step >= self.stop_steps:
                    self.dumplog(
                        "The best solution has not "
                        "improved for {} steps\n".format(self.no_improvement_step),
                        highlight="#",
                    )
                    raise RuntimeError("The best solution is not improved for {} steps".format(self.no_improvement_step))
                self.dumplog("-------------------------------------------------------")
                time_label = get_current_time()
                self.dumplog(
                    "%s:  Starting Basin-Hopping Step %05d\n" % (time_label, self.nsteps)
                )
                
                for number_of_trials in range(maximum_trial):
                    modifier_name = self.select_modifier()
                    self.modifier_name = modifier_name
                    try:
                        new_atoms = []
                        for _ in range(n_multiple):
                            new_atoms.append(self.move(modifier_name=modifier_name))
                    except (
                        NoReasonableStructureFound
                    ) as emsg:  # emsg stands for error message
                        if not isinstance(emsg, str):
                            emsg = "Unknown"
                        self.dumplog(
                            "%s did not find a good structure because of %s"
                            % (modifier_name, emsg)
                        )
                    else:
                        self.on_optimization = self.nsteps
                        self.dumplog(
                            f"{len(new_atoms)} structure found, begin to optimize this structure\n"
                        )
                        # self.log_status()
                        self.save_current_status()  # before optimization switch on the self.on_optimization flag
                        # self.dumplog("{}: begin structure optimization subroutine".format(get_current_time()))
                        new_atoms = self.optimize(inatoms=new_atoms, multiple=multiple)
                        # self.dumplog("{}: Optimization Done\n".format(get_current_time()))
                        self.accepting_new_structures(
                            newatoms=new_atoms, move_action=modifier_name
                        )
                        self.on_optimization = -1  # switch off the optimization status
                        # self.log_status()
                        self.save_current_status()
                        self.nsteps += 1
                        break
                else:
                    raise RuntimeError(
                        "Program does not find a good structure after {} tests".format(
                            maximum_trial
                        )
                    )
        else:
            while self.nsteps < maximum_steps:
                if self.no_improvement_step >= self.stop_steps:
                    self.dumplog(
                        "The best solution has not "
                        "improved for {} steps\n".format(self.no_improvement_step),
                        highlight="#",
                    )
                    raise RuntimeError("The best solution is not improved for {} steps".format(self.no_improvement_step))
                self.dumplog("-------------------------------------------------------")
                time_label = get_current_time()
                self.dumplog(
                    "%s:  Starting Basin-Hopping Step %05d\n" % (time_label, self.nsteps)
                )
    
                for number_of_trials in range(maximum_trial):
                    modifier_name = self.select_modifier()
                    self.modifier_name = modifier_name
                    try:
                        new_atoms = self.move(modifier_name=modifier_name)
                    except (
                        NoReasonableStructureFound
                    ) as emsg:  # emsg stands for error message
                        if not isinstance(emsg, str):
                            emsg = "Unknown"
                        self.dumplog(
                            "%s did not find a good structure because of %s"
                            % (modifier_name, emsg)
                        )
                    else:
                        self.on_optimization = self.nsteps
                        self.dumplog(
                            "One structure found, begin to optimize this structure\n"
                        )
                        # self.log_status()
                        self.save_current_status()  # before optimization switch on the self.on_optimization flag
                        # self.dumplog("{}: begin structure optimization subroutine".format(get_current_time()))
                        self.optimize(inatoms=new_atoms)
                        # self.dumplog("{}: Optimization Done\n".format(get_current_time()))
                        self.accepting_new_structures(
                            newatoms=new_atoms, move_action=modifier_name
                        )
                        self.on_optimization = -1  # switch off the optimization status
                        # self.log_status()
                        self.save_current_status()
                        self.nsteps += 1
                        break
                else:
                    raise RuntimeError(
                        "Program does not find a good structure after {} tests".format(
                            maximum_trial
                        )
                    )

    # def natural_cutoffs(self, atoms, multiplier=1.1):
    #     """Generate a neighbor list cutoff for every atom"""
    #     return [covalent[atom.number] * multiplier for atom in atoms]

    def examine_unconnected_components(self, newatoms):
        nat_cut = natural_cutoffs(newatoms, mult=1.2)
        nl = NeighborList(nat_cut, skin=0, self_interaction=False, bothways=True)
        nl.update(newatoms)
        matrix = nl.get_connectivity_matrix()
        n_components, component_list = sparse.csgraph.connected_components(matrix)
        self.dumplog("There are {} components in the system".format(n_components))
        if n_components == 1:
            return True, n_components
        elif n_components > 1:
            return False, n_components
        
    def examine_water_molecule_presents(self, newatoms):
        nat_cut = natural_cutoffs(newatoms, mult=1.25)
        nl = NeighborList(nat_cut, skin=0, self_interaction=False, bothways=True)
        nl.update(newatoms)
        matrix = nl.get_connectivity_matrix()
        surf_ind = []
        for a in newatoms:
            surf_ind.append(a.index)
        water = []
        for i in surf_ind:
            if newatoms[i].symbol == 'O':
                indices, offsets = nl.get_neighbors(i)
                near_H = []
                for a in indices:
                    if newatoms[a].symbol == 'H':
                        near_H.append(a)
                # print(near_H)
                if len(near_H) >= 2:
                    molc_water = True
                    for n in near_H:
                        test, _ = nl.get_neighbors(n)
                        # if H in the water molecule has a bond with carbon in molcule, neglect
                        if any(newatoms[t].symbol == 'C' for t in test):
                            molc_water = False
                            pass
                        else:
                            water.append(n)
                    if molc_water:
                        water.append(i)

        self.dumplog("There are {} number of water molecules in the system".format(len(water)/3))
        if len(water) > 0:
            return True, water
        else:
            return False, None

    def accepting_new_structures(self, newatoms=None, move_action=None):
        """This function takes care of all the accepting algorithm. I.E metropolis algorithms
        newatoms is the newly optimized structure
        move_action is action (modifier name) to  produce the initial structure for newatoms;
        If move_action is specified, its weights will be adjusted according to the acception or rejection; otherwise,
        the weights are not altered"""

        assert newatoms is not None

        self.opt_trajectory.write(newatoms)

        En = newatoms.get_potential_energy()  # Energy_new
        Fn = En - self.get_ref_potential(newatoms)  # Free_energy_new

        accept = False
        modifier_weight_action = "decrease"
        water_presents, _ = self.examine_water_molecule_presents(newatoms)
        connected, n_components = self.examine_unconnected_components(newatoms)
        if Fn < self.free_energy and connected and not water_presents:
            accept = True
            modifier_weight_action = "increase"
        elif Fn < self.free_energy and n_components <= 2 and not water_presents:
            self.dumplog("There are {} number of components in the system\n".format(n_components))
            accept = True
            modifier_weight_action = "increase"
        elif np.random.uniform() < np.exp(-(Fn - self.free_energy) / self.T / units.kB) and connected and not water_presents:
            accept = True
        elif np.random.uniform() < np.exp(-(Fn - self.free_energy) / self.T / units.kB) and n_components <= 2 and not water_presents:
            self.dumplog("There are {} number of components in the system\n".format(n_components))
            accept = True

        if move_action is not None:
            self.update_modifier_weights(
                name=move_action, action=modifier_weight_action
            )

        if accept:
            _int_accept = 1
            self.dumplog("Accepted, F(old)=%.3f F(new)=%.3f\n" % (self.free_energy, Fn))
            self.update_self_atoms(newatoms)
            self.energy = En
            self.free_energy = Fn
        else:
            _int_accept = 0
            self.dumplog("Rejected, F(old)=%.3f F(new)=%.3f\n" % (self.free_energy, Fn))
            if not connected:
                self.dumplog("{} atoms have migrated out of the surface.".format(n_components))
            self.rejected_steps += 1

        if accept:
            self.lm_trajectory.write(self.atoms, accept=1)
        else:
            self.lm_trajectory.write(self.atoms, accept=0)

        # adjust the temperatures
        self.accept_history.append(_int_accept)
        self.accept_history_full.append(_int_accept)
        if len(self.accept_history) > self.max_history:
            self.accept_history.pop(0)
            _balance = sum(self.accept_history) / float(self.max_history)
            if _balance > 2.0 * (1 - _balance):
                self.T = self.T / 1.03
            elif _balance < 0.5 * (1 - _balance):
                self.T = self.T * 1.03

        if self.T < self.min_T:
            self.T = self.min_T
        elif self.T > self.max_T:
            self.T = self.max_T

        # update the best result for this basin-hopping
        if self.free_energy < self.free_energy_min:
            self.free_energy_min = self.free_energy
            self.no_improvement_step = 0
        else:
            self.no_improvement_step += 1

        # self.energy is not used for updating no_improvement_step
        if self.energy < self.energy_min:
            self.energy_min = self.energy

        # self.log_status()
        self.save_current_status()
        self.log_status()
        self.dumplog("-------------------------------------------------------")
    
    # def optimize_script(self, inatoms=None):
    #     atoms = inatoms.copy()
    #     atoms.pbc = True
    #     pos = atoms.get_positions()
    #     posz = pos[:, 2]
    #     posz_min = np.min(posz)
    #     posz_mid = posz_min + 5
    #     if not self.cmds:
    #         el = []
    #         uniq_elements = np.unique(atoms.get_chemical_symbols())
    #         for e in uniq_elements:
    #             el.append(self.elements[e])
    #         self.el = ' '.join(map(str, np.sort(el)[::-1]))
    #         mass = []
    #         for e in np.sort(el)[::-1]:
    #             mass.append(atomic_masses[e])
    #         self.mass = ' '.join(map(str, mass))
    #         self.cmds = ['pair_style quip',
    #                      'pair_coeff * * {} "Potential xml_label={}" {}'.format(self.model_file, self.model_label, self.el),
    #                      'region slab block EDGE EDGE EDGE EDGE 0 {}'.format(posz_mid),
    #                      'group fixed_slab region slab', 
    #                      'fix freeze fixed_slab setforce 0.0 0.0 0.0',
    #                      'dump 1 all custom 1 md.lammpstrj id type x y z vx vy vz fx fy fz',
    #                      'thermo 1',
    #                      'thermo_style custom step fmax press cpu ke pe etotal temp',
    #                      'min_style cg',
    #                      'minimize 0.0 1.0e-4 200 1000000',]
    #         self.lammps = LAMMPSlib(lmpcmds=self.cmds, log_file='out')
    #     else:
    #         el = []
    #         uniq_elements = np.unique(atoms.get_chemical_symbols())
    #         for e in uniq_elements:
    #             el.append(self.elements[e])
    #         el = ' '.join(map(str, np.sort(el)[::-1]))
    #         if self.el != el:
    #             self.el = el
    #             self.cmds = ['pair_style quip',
    #                      'pair_coeff * * {} "Potential xml_label={}" {}'.format(self.model_file, self.model_label, self.el),
    #                      'region slab block EDGE EDGE EDGE EDGE 0 {}'.format(posz_mid),
    #                      'group fixed_slab region slab', 
    #                      'fix freeze fixed_slab setforce 0.0 0.0 0.0',
    #                      'dump 1 all custom 1 md.lammpstrj id type x y z vx vy vz fx fy fz',
    #                      'thermo 1',
    #                      'thermo_style custom step fmax press cpu ke pe etotal temp',
    #                      'min_style cg',
    #                      'minimize 0.0 1.0e-4 200 1000000',]
    #             self.lammps = LAMMPSlib(lmpcmds=self.cmds, log_file='out')
    #         else:
    #             pass

        
    #     ndx = np.where(posz < posz_mid)[0]
    #     c = FixAtoms(ndx)
    #     atoms.set_constraint(c)

    #     atoms.calc = self.lammps
    #     bfgs = BFGS(atoms, logfile='stdout')
    #     traj = Trajectory('opt.traj', 'w', atoms)
    #     bfgs.attach(traj)
    #     bfgs.run(fmax=0.01)

    #     atoms = read("opt.traj@-1")
    #     e = atoms.get_potential_energy()
    #     f = atoms.get_forces()
    #     atoms.set_calculator(SinglePointCalculator(atoms, energy=e, forces=f))
    #     atoms.write("optimized.traj")
    #     # return atoms

    def optimize(self, inatoms=None, restart=False, multiple=False):
        self.dumplog(
            "{}: begin structure optimization subroutine at step {}".format(
                get_current_time(), self.nsteps
            )
        )
        if multiple:
            atoms = inatoms.copy()
            opt_dir = self.opt_folder
            steps = self.nsteps
            script = self.bash_script
            copied_files = self.copied_files[:]
            topdir = os.getcwd()
            subdir = os.path.join(topdir, opt_dir, "opt_%05d" % steps)
            if restart:
                assert os.path.isdir(subdir)
            else:
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)
                # prepare all the files in the subfolders
                if script not in copied_files and not self.model_file:
                    copied_files.append(script)
                for fn in copied_files:
                    assert os.path.isfile(fn)
                    shutil.copy(os.path.join(topdir, fn), os.path.join(subdir, fn))
                write(os.path.join(subdir, "input.traj"), atoms)
            try:
                os.chdir(subdir)
                if not self.model_file:
                    opt_job = subprocess.Popen(["bash", script], cwd=subdir)
                    opt_job.wait()
                    if opt_job.returncode < 0:
                        sys.stderr.write(
                            "optimization does not terminate properly at {}".format(subdir)
                        )
                        sys.exit(1)
                else:
                    self.optimize_script(atoms)
            except:
                raise RuntimeError(
                    "some error encountered at folder {} during optimizations".format(
                        subdir
                    )
                )
            else:
                fn = os.path.join(subdir, "optimized.traj")
                assert os.path.isfile(fn)
                optimized_atoms = read(fn)
            finally:
                os.chdir(topdir)
    
            e = optimized_atoms.get_potential_energy()
            f = optimized_atoms.get_forces()
            # set new positions for atoms
            # constraints_list = []
            # # all the FixAtoms constraints and Hookean constraints are passed over
            # for c in optimized_atoms.constraints:
            #     if isinstance(c, FixAtoms):
            #         constraints_list.append(c)
            #     elif isinstance(c, Hookean):
            #         constraints_list.append(c)
            # cell = optimized_atoms.get_cell()
            # pbc = optimized_atoms.get_pbc()
            # inatoms = inatoms[0]
            # inatoms.set_constraint()
            # del inatoms[range(inatoms.get_global_number_of_atoms())]
            # inatoms.extend(optimized_atoms)
            # inatoms.set_pbc(pbc)
            # inatoms.set_cell(cell)
            # inatoms.set_constraint(optimized_atoms.constraints)
            inatoms = optimized_atoms
            spc = SinglePointCalculator(inatoms, energy=e, forces=f)
            inatoms.set_calculator(spc)
            self.dumplog("{}: Optimization Done\n".format(get_current_time()))            
            return inatoms
        else:
            atoms = inatoms.copy()
            opt_dir = self.opt_folder
            steps = self.nsteps
            script = self.bash_script
            copied_files = self.copied_files[:]
            topdir = os.getcwd()
            subdir = os.path.join(topdir, opt_dir, "opt_%05d" % steps)
            if restart:
                assert os.path.isdir(subdir)
            else:
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)
                # prepare all the files in the subfolders
                if script not in copied_files and not self.model_file:
                    copied_files.append(script)
                for fn in copied_files:
                    assert os.path.isfile(fn)
                    shutil.copy(os.path.join(topdir, fn), os.path.join(subdir, fn))
                write(os.path.join(subdir, "input.traj"), atoms)
            try:
                os.chdir(subdir)
                if not self.model_file:
                    opt_job = subprocess.Popen(["bash", script], cwd=subdir)
                    opt_job.wait()
                    if opt_job.returncode < 0:
                        sys.stderr.write(
                            "optimization does not terminate properly at {}".format(subdir)
                        )
                        sys.exit(1)
                else:
                    self.optimize_script(atoms)
            except:
                raise RuntimeError(
                    "some error encountered at folder {} during optimizations".format(
                        subdir
                    )
                )
            else:
                fn = os.path.join(subdir, "optimized.traj")
                assert os.path.isfile(fn)
                optimized_atoms = read(fn)

                water_presents, water = self.examine_water_molecule_presents(optimized_atoms)
                while water_presents:
                    self.dumplog("Remove water molecules present and restart optimizations\n")
                    del optimized_atoms[[atom.index for atom in optimized_atoms if atom.index in water]]
                    write(os.path.join(subdir, "input.traj"), optimized_atoms)
                    try:
                        if not self.model_file:
                            opt_job = subprocess.Popen(["bash", script], cwd=subdir)
                            opt_job.wait()
                            if opt_job.returncode < 0:
                                sys.stderr.write(
                                    "optimization does not terminate properly at {}".format(subdir)
                                )
                                sys.exit(1)
                        else:
                            self.optimize_script(atoms)
                    except:
                        raise RuntimeError(
                            "some error encountered at folder {} during optimizations".format(
                                subdir
                            )
                        )
                    else:
                        fn = os.path.join(subdir, "optimized.traj")
                        assert os.path.isfile(fn)
                        optimized_atoms = read(fn)
                        water_presents, water = self.examine_water_molecule_presents(optimized_atoms)
                self.dumplog("No water molecule remaining\n")
            finally:
                os.chdir(topdir)
    
            e = optimized_atoms.get_potential_energy()
            f = optimized_atoms.get_forces()
            # set new positions for atoms
            # constraints_list = []
            # # all the FixAtoms constraints and Hookean constraints are passed over
            # for c in optimized_atoms.constraints:
            #     if isinstance(c, FixAtoms):
            #         constraints_list.append(c)
            #     elif isinstance(c, Hookean):
            #         constraints_list.append(c)
            cell = optimized_atoms.get_cell()
            pbc = optimized_atoms.get_pbc()
            inatoms.set_constraint()
            del inatoms[range(inatoms.get_global_number_of_atoms())]
            inatoms.extend(optimized_atoms)
            inatoms.set_pbc(pbc)
            inatoms.set_cell(cell)
            inatoms.set_constraint(optimized_atoms.constraints)
            spc = SinglePointCalculator(inatoms, energy=e, forces=f)
            inatoms.set_calculator(spc)
            self.dumplog("{}: Optimization Done\n".format(get_current_time()))

    def get_ref_potential(self, atoms=None):
        """
        calculate the chemical potential of atoms
        :param atoms:
        :return:
        """
        ref = 0.0
        for i, si in enumerate(atoms.get_chemical_symbols()):
            if si not in self.mu.keys():
                raise RuntimeError(
                    "I did not find the chemical potential for element %s" % si
                )
            else:
                ref += self.mu.get(si)
        return ref

    def update_self_atoms(self, a):
        """
        This function will keep the original reference of self.atoms, but refresh it with new structures.
        You have to keep the reference of self.atoms, otherwise, self.call_observers will not work.
        :param a: ase.atoms.Atoms object.
        :return: None
        """
        self.atoms.set_constraint()
        del self.atoms[range(self.atoms.get_global_number_of_atoms())]
        cell = a.get_cell()
        pbc = a.get_pbc()
        self.atoms.extend(a.copy())
        self.atoms.set_pbc(pbc)
        self.atoms.set_cell(cell)
        self.atoms.set_constraint(a.constraints)
        try:
            e = a.get_potential_energy()
            f = a.get_forces()
        except PropertyNotImplementedError:
            self.dumplog("Warnning : self.atoms no energy !!!!")
        else:
            spc = SinglePointCalculator(self.atoms, forces=f, energy=e)
            self.atoms.set_calculator(spc)
