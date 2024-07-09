from dataclasses import dataclass, field
from typing import Literal, NamedTuple
from pathlib import Path
import json
import subprocess
import os

import numpy as np
import numpy.typing as npt
from PIL import Image as PilImage
from skimage.io import imsave
import matplotlib.pyplot as plt

# afa.make_files.get_source_files
# afa.get_landmarks
# afa.align_procrustes

# import alignfaces as afa
# import .align as afa
from .align.make_aligned_faces import get_source_files, get_landmarks, align_procrustes, place_aperture
from .align.make_aligned_faces import morph_between_two_faces as morph
from .align.make_files import clone_directory_tree

from .features.filters import face_filters, get_eo, eo_montage
from .features.roi import get_feature_roi
from .features.coefficients import select_coefficients_by_feature
from .features.reconstruct import wrff_progress as reconstruct
from .features.reconstruct import combine_features
from .features.shift_features import thatcher_face, space_out_features, space_out_ipd, double_face_illusion



# source_error = prepare_source(ens, pre=pre, post=post)
# aligned_error = prepare_aligned(ens, pre=pre, post=post, ipd=64)
# features_error = prepare_features(ens, pre=pre, post=post, make_all=True)
# make_features(ens, "richard-pryor.jpg")
# for f in list(ens.landmarks):
#     make_features(ens, f)

class FileBookends(NamedTuple):
    prefix: str
    postfix: Literal["png", "jpg","jpeg", "tiff", "tif"]

NDArrayImage = npt.NDArray[np.uint8]
NDArrayFloat = npt.NDArray[np.float64]
NDArrayComplex = npt.NDArray[np.complex128] # NDArrayComplex = npt.NDArray[np.complex_]

@dataclass
class Face:
    """Specific face image."""
    name: str
    img: NDArrayImage       # face image
    ROI: NDArrayFloat       # spatial map per feature {0.,1.}
    F: NDArrayImage         # features
    SHAPE: tuple[int, int]  # pixels (height, width)
    feature_labels: list[str]

@dataclass
class Ensemble:
    """Ensemble of aligned face images."""
    NUM_FEATURES = 6
    feature_labels = ['left_eyebrow', 'right_eyebrow', 'left_eye', 'right_eye', 'nose', 'mouth_outline']

    dir_source: Path
    file_bookends: FileBookends
    make_all_features: bool = False
    make_windowed_faces: bool = False
    dir_aligned: Path | None = None
    dir_windowed: Path | None = None
    dir_wrff: Path | None = None
    SOURCE: bool  | None = None
    ALIGNED: bool  | None = None
    WINDOW: bool | None = None
    WRFF: bool | None = None
    adjust_size: Literal['set_eye_distance', 'set_image_width','set_image_height', 'default'] | None = None
    SHAPE: tuple[int, int]  | None = None       # pixels (height, width)
    INTER_PUPILLARY_DISTANCE: int | None = None # pixels
    
    aperture: NDArrayFloat | None = None

    # Required for WRFF generation
    landmarks: dict | None = None
    ffilters: NDArrayFloat | None = None    # Fourier filters, dtype('float64')
    sfilters: NDArrayComplex | None = None  # Spatial filters, dtype('complex128')
    CriterionQuantileAmp: float = .5        # proportion of wavelets selected (per feature)    
    # file_extension: Literal["png", "jpg","jpeg", "tiff", "tif"] | None = None

    # Current roster of faces to work with
    Roster: list[Face] = field(default_factory=list)

    # Initialize Ensemble
    def __post_init__(self):
        self.file_extension = str(self.file_bookends[1])
        self.dir_aligned = self.dir_source.parent / (self.dir_source.stem + "-aligned")
        self.dir_windowed = self.dir_source.parent / (self.dir_source.stem + "-aligned-windowed")
        self.dir_wrff = self.dir_source.parent / (self.dir_source.stem + "-aligned-wrff")
        source_error = prepare_source(self)
        if source_error == None:
            self.SOURCE = True
            aligned_error = prepare_aligned(self)
            if aligned_error == None:
                self.ALIGNED = True    
                features_error = prepare_features(self)
                if features_error == None:
                    self.WRFF = True
                else:
                    print(features_error)
                    self.WRFF = False
            else:
                print(aligned_error)
                self.ALIGNED = False
                self.WRFF = False
                self.WINDOW = False
        else: 
            print(source_error)
            self.SOURCE = False
            self.ALIGNED = False
            self.WRFF = False
            self.WINDOW = False
        # if self.dir_aligned.exists():
        #     self.ALIGNED = True
        # else:
        #     self.ALIGNED = False
        # if self.dir_wrff.exists():
        #     self.WRFF = True
        # else:
        #     self.WRFF = False
        # DO -- run landmarks for source if none
        # DO -- run alignment if none
        # DO -- run landmarks for aligned if none
        if self.ALIGNED:
            # Landmarks
            if (self.dir_aligned / "landmarks.txt").exists():
                with open(self.dir_aligned / "landmarks.txt") as json_data:
                    self.landmarks = json.load(json_data)
                    json_data.close()
                self.file_extension = list(self.landmarks)[0].split(".")[1]
                # Constants of aligned-image ensemble
                if (self.dir_aligned / "specs.csv").exists():
                    csvfile = open(self.dir_aligned / "specs.csv", "r")
                    _ = csvfile.readline()
                    values = csvfile.readline()
                    self.adjust_size, h, w, IPDtext = [v for v in values.split(",")]
                    h = int(h)
                    w = int(w)
                    self.INTER_PUPILLARY_DISTANCE = int(IPDtext.split("\n")[0])
                    self.SHAPE = (h, w)
                    results = face_filters(self.SHAPE, self.INTER_PUPILLARY_DISTANCE)
                    self.ffilters = results["ffilters"]
                    self.sfilters = results["sfilters"]
            else:
                    print("Extract landmarks from dir_aligned.")
        else:
            if(self.dir_source / "landmarks.txt").exists():
                with open(self.dir_source / "landmarks.txt") as json_data:
                    self.landmarks = json.load(json_data)
                    json_data.close()
                self.file_extension = list(self.landmarks)[0].split(".")[1]
            else:
                print("Extract landmarks from dir_source.")
                print("Run alignment on dir_source.")
                print("Extract landmarks from dir_aligned.")
        if not self.WRFF:
            print("Batch WRFF.")

    # Ensemble methods ----------------------------------------------

    # Ensemble management
    def list_faces(self):
        print("\n\nFaces in ensemble")
        for f in list(self.landmarks):
            print("\t" + f)        
        print("\n\n")    

    def get_face_list(self) -> list:
        return list(self.landmarks)
    
    # Roster methods ------------------------------------------------

    # Roster management
    def add_to_roster(self, file_name):
        if len(file_name.split(".")) == 1:
            file_name = file_name + "." + self.file_extension
        L = self.landmarks[file_name]
        if self.ALIGNED:
            results = get_feature_roi(self.SHAPE, L)
            ROI = results["DilatedBW"]            
            img = np.array(PilImage.open(self.dir_aligned / file_name).convert("L"))    
            ID = file_name.split(".")[0]
            file_full_1 = file_full = self.dir_wrff / (ID + "-1.png")
            if self.WRFF and file_full_1.exists():
                # ID = file_name.split(".")[0]
                R, C = self.SHAPE
                F = np.zeros((R,C,6), dtype=np.uint8)
                for i in range(6):
                    file_full = self.dir_wrff / (ID + "-" + str(i+1) + ".png")
                    F[:,:,i] = np.array(PilImage.open(file_full).convert("L"))
            else:
                EO = get_eo(img, self.ffilters)
                chosen_coefficients = select_coefficients_by_feature(EO, ROI, self.CriterionQuantileAmp)
                F = reconstruct(self.sfilters, EO, chosen_coefficients, img_format="uint8")
                # 🚧 write to disk
                for i in range(6):
                    file_full = self.dir_wrff / (ID + "-" + str(i+1) + ".png")
                    imsave(file_full, F[:,:,i], check_contrast=False)
                self.WRFF = True
            SHAPE = F.shape[:2]
        else:
            img = np.array(PilImage.open(self.dir_source / file_name).convert("L"))
            F = None
            SHAPE = img.shape
            ROI = None
        self.Roster.append(Face(name=file_name,img=img,ROI=ROI,F=F,SHAPE=SHAPE,feature_labels=self.feature_labels))

    def add_all_to_roster(self):
        file_names = list(self.landmarks)
        for f in file_names:
            self.add_to_roster(f)

    def empty_roster(self):
        self.Roster = []

    # Roster checks
    def roster_exists(self):
        return(len(self.Roster) > 0)
        
    def roster_same_shapes(self):
        shapes = [face.F.shape for face in self.Roster]
        return(len(set(shapes))==1)

    def roster_same_labels(self):
        labels = [tuple(face.feature_labels) for face in self.Roster]
        return(len(set(labels))==1)
    
    def roster_features_separate(self):
        return(len(self.Roster[0].F.shape) == 3)

    # Roster utility
    def get_roster_feature_index(self):
        if not self.roster_same_labels():
            print("Roster labels vary")
        else:
            labels = self.Roster[0].feature_labels
            feature_index = {}
            for i, l in enumerate(labels):
                feature_index[l] = i
            return feature_index
    
    def clip_roster_margins_(self, margins=(1/6, 1/4)):  # {F, ROI}
        """Limited utility: Clipping relative to individual ROI."""
        for fi, face in enumerate(self.Roster):
            name = face.name
            img = face.img
            F = face.F
            ROI = face.ROI
            SHAPE = face.SHAPE
            feature_labels = face.feature_labels

            vertical_extent = np.where(ROI.sum((1,2)))[0][[0,-1]]
            horizontal_extent = np.where(ROI.sum((0,2)))[0][[0,-1]]
            x0 = round(horizontal_extent[0] - np.diff(horizontal_extent)[0]*margins[0])
            x1 = round(horizontal_extent[1] + np.diff(horizontal_extent)[0]*margins[0])
            y0 = round(vertical_extent[0] - np.diff(vertical_extent)[0]*margins[1])
            y1 = round(vertical_extent[1] + np.diff(vertical_extent)[0]*margins[1])
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(SHAPE[1], x1)
            y1 = min(SHAPE[0], y1)
            F_ = F[y0:y1, x0:x1, :]
            ROI_ = ROI[y0:y1, x0:x1, :]
            SHAPE_ = F_.shape[:2]
            img_ = img[y0:y1, x0:x1]
            face_ = Face(name=name,img=img_,ROI=ROI_,F=F_,SHAPE=SHAPE_,feature_labels=feature_labels)
            self.Roster[fi] = face_

    def clip_roster_margins(self, margins=(1/6, 1/4)):  # {F, ROI}
        """More widely appicable: Clipping relative to union of ROI in Roster."""
        if not self.roster_exists():
            print("Roster is empty.")
        elif not self.roster_features_separate():
            print("Roster features already combined.")
        elif not self.roster_same_shapes():
            print("Shapes vary in Roster. Using clip_roster_margins_() instead.")
            self.clip_roster_margins_(margins=margins)
        else:
            tuple_of_roi = tuple( [face.ROI[:,:,:,np.newaxis] for face in self.Roster] )
            ROIS = np.concatenate(tuple_of_roi, axis=3)
            ROI = ROIS.sum(axis=3) > 0
            vertical_extent = np.where(ROI.sum((1,2)))[0][[0,-1]]
            horizontal_extent = np.where(ROI.sum((0,2)))[0][[0,-1]]
            x0 = round(horizontal_extent[0] - np.diff(horizontal_extent)[0]*margins[0])
            x1 = round(horizontal_extent[1] + np.diff(horizontal_extent)[0]*margins[0])
            y0 = round(vertical_extent[0] - np.diff(vertical_extent)[0]*margins[1])
            y1 = round(vertical_extent[1] + np.diff(vertical_extent)[0]*margins[1])
            x0 = max(0, x0)
            y0 = max(0, y0)            
            for fi, face in enumerate(self.Roster):
                name = face.name
                img = face.img
                F = face.F
                SHAPE = face.SHAPE
                feature_labels = face.feature_labels
                x1 = min(SHAPE[1], x1)
                y1 = min(SHAPE[0], y1)
                F_ = F[y0:y1, x0:x1, :]
                ROI_ = ROI[y0:y1, x0:x1, :]
                SHAPE_ = F_.shape[:2]
                img_ = img[y0:y1, x0:x1]
                face_ = Face(name=name,img=img_,ROI=ROI_,F=F_,SHAPE=SHAPE_,feature_labels=feature_labels)
                self.Roster[fi] = face_

    def combine_roster_features(self):
        if not self.roster_exists():
            print("Roster is empty.")
        elif not self.roster_features_separate():
            print("Roster features already combined.")
        else:
            for fi, face in enumerate(self.Roster):
                name = face.name
                img = face.img
                F = face.F
                ROI = face.ROI
                SHAPE = face.SHAPE     
                feature_labels=face.feature_labels   
                F = combine_features(F)
                face_ = Face(name=name,img=img,ROI=ROI,F=F,SHAPE=SHAPE,feature_labels=feature_labels)
                self.Roster[fi] = face_

    def display_roster(self, include="both", show=True, title=True):
        assert include in ["both","source","features"]
        if (self.ALIGNED == False) or (self.WRFF == False):
            include = "source"
        fh = []
        for fi, face in enumerate(self.Roster):
            name = face.name
            img = face.img
            F = face.F
            ROI = face.ROI
            SHAPE = face.SHAPE
            if (include=="both") or (include=="features"):
                if len(F.shape) > 2:   
                    F = combine_features(F)
            if include == "both" and self.roster_same_shapes():
                montage = np.c_[img, F]
            elif include == "source":
                montage = img
            else:
                montage = F
            fig, ax = plt.subplots()
            ax.imshow(montage, cmap="gray")
            ax.tick_params(left = False, right = False , labelleft = False , 
                                        labelbottom = False, bottom = False)
            if title:
                plt.title(name)
            fh.append(fig)
            if show:
                plt.show()
        return fh

    # Roster effects
    def roster_thatcher(self):
        feature_index = self.get_roster_feature_index()
        for fi, face in enumerate(self.Roster):
            name = face.name
            img = face.img
            F = face.F
            ROI = face.ROI
            SHAPE = face.SHAPE                    
            feature_labels = face.feature_labels
            results = thatcher_face(F, ROI, feature_index)
            Ft, ROIt = results["F"], results["ROI"]
            face_ = Face(name=name,img=img,ROI=ROIt,F=Ft,SHAPE=SHAPE,feature_labels=feature_labels)
            self.Roster[fi] = face_

    def roster_space_out(self, scale, scale_shape=None):
        if not self.roster_exists():
            print("Roster is empty.")
        elif not self.roster_features_separate():
            print("Roster features already combined.")
        else:
            if scale_shape is None:
                scale_shape = scale
            assert scale_shape >= scale
            for fi, face in enumerate(self.Roster):
                name = face.name
                img = face.img
                F = face.F
                ROI = face.ROI
                feature_labels = face.feature_labels
                results = space_out_features(F, ROI, scale=scale, scale_shape=scale_shape)
                SOF, SOR = results["F"], results["ROI"]
                SHAPE_ = SOF.shape[:2]
                face_ = Face(name=name,img=img,ROI=SOR,F=SOF,SHAPE=SHAPE_,feature_labels=feature_labels)
                self.Roster[fi] = face_

    def roster_space_ipd(self, scale):
        if not self.roster_exists():
            print("Roster is empty.")
        elif not self.roster_features_separate():
            print("Roster features already combined.")
        else:
            for fi, face in enumerate(self.Roster):
                name = face.name
                img = face.img
                F = face.F
                ROI = face.ROI
                feature_labels = face.feature_labels
                results = space_out_ipd(F, ROI, scale=scale)
                SOF, SOR = results["F"], results["ROI"]
                SHAPE_ = SOF.shape[:2]
                face_ = Face(name=name,img=img,ROI=SOR,F=SOF,SHAPE=SHAPE_,feature_labels=feature_labels)
                self.Roster[fi] = face_        

    def roster_double_face(self, percent_down_shift=None):
        """🚧 checks on number of features in other roster functions 🚧"""
        if not self.roster_exists():
            print("Roster is empty.")
        elif not self.roster_features_separate():
            print("Roster features already combined.")
        elif self.Roster[0].F.shape[2] > 6:
            print("Roster features already doubled.")
        else:
            if percent_down_shift is None:
                percent_down_shift = .92
            feature_index = self.get_roster_feature_index()
            for fi, face in enumerate(self.Roster):             
                name = face.name
                img = face.img
                F = face.F
                ROI = face.ROI
                SHAPE = face.SHAPE
                feature_labels = face.feature_labels
                results = double_face_illusion(F, ROI, feature_index, percent_down_shift=percent_down_shift)
                DF, DR, FL = results["F"], results["ROI"], results["feature_labels"]
                face_ = Face(name=name,img=img,ROI=DR,F=DF,SHAPE=SHAPE,feature_labels=FL)
                self.Roster[fi] = face_

    def roster_chimeric(self, feature_id):
        if not self.roster_exists():
            print("Roster is empty.")
        elif not self.roster_features_separate():
            print("Roster features already combined.")
        elif not self.roster_same_shapes():
            print("Roster shapes vary.")
        elif not self.roster_same_labels():
            print("Roster labels vary.")
        else:
            assert type(feature_id) is dict
            assert self.feature_labels ==list(feature_id)
            v = list(feature_id.values())
            assert max(v) < len(self.Roster)
            assert min(v) >= 0
            ROI, F = [], []
            for face in self.Roster:
                ROI.append(face.ROI)
                F.append(face.F)
            feature_index = self.get_roster_feature_index()
            C = np.zeros_like(self.Roster[0].F)
            R = np.zeros_like(C)
            for fi in feature_index:
                f = F[feature_id[fi]][:,:,feature_index[fi]]
                C[:,:,feature_index[fi]] = f
                r = ROI[feature_id[fi]][:,:,feature_index[fi]]
                R[:,:,feature_index[fi]] = r
            img = combine_features(C)
            return {"img": img, "F": C, "ROI": R}

        #double_face_illusion    
    # NEXT: method to set CriterionQuantileAmp
    # NEXT: method to show face
    # NEXT: method to add Face to Faces

    def roster_morph_between(self, num_morphs: int):
        assert len(self.Roster)==2
        assert num_morphs > 2
        face_array, p, morph_path = morph(str(self.dir_aligned) + os.sep,
                                        do_these=[0, 1],
                                        num_morphs=num_morphs,
                                        file_prefix=self.file_bookends[0],
                                        file_postfix=self.file_bookends[1])
        results = {"face_array": face_array, "p": p, "morph_path": morph_path}
        return results

def make_features(ens: Ensemble, file_name: str):
    L = ens.landmarks[file_name]
    results = get_feature_roi(ens.SHAPE, L)
    ROI = results["DilatedBW"]
    img = np.array(PilImage.open(ens.dir_aligned / file_name).convert("L"))    
    EO = get_eo(img, ens.ffilters)
    chosen_coefficients = select_coefficients_by_feature(EO, ROI, ens.CriterionQuantileAmp)
    F = reconstruct(ens.sfilters, EO, chosen_coefficients, img_format="uint8")
    ID = file_name.split(".")[0]
    for i in range(6):
        file_full = ens.dir_wrff / (ID + "-" + str(i+1) + ".png")
        imsave(file_full, F[:,:,i], check_contrast=False)        

def prepare_source(ens: Ensemble):
    pre=str(ens.file_bookends[0])
    post=str(ens.file_bookends[1])
    source_error = "Unknown"
    if not ens.dir_source.exists():
        source_error = "Source folder does not exist."
    else:
        # source_files = afa.make_files.get_source_files(ens.dir_source, pre, post)
        source_files = get_source_files(ens.dir_source, pre, post)
        if len(source_files) == 0:
            source_error = "No files in source folder matching requested pattern."
        else:
            landmarks_file = ens.dir_source / "landmarks.txt"
            if not landmarks_file.exists():
                print("Missing file: landmarks.txt")
                print("Get landmarks using AFA.")
                input_path = str(ens.dir_source) + os.sep
                # afa.get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)
                get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)
                source_error = None
            else:
                with open(ens.dir_source / "landmarks.txt") as json_data:
                    landmarks_source = json.load(json_data)
                    json_data.close()
                ens.file_extension = list(landmarks_source)[0].split(".")[1]
                if ens.file_extension != post:
                    print("Image files with landmarks have different file extension than specified.")
                    print("Get landmarks using AFA.")
                    input_path = str(ens.dir_source) + os.sep
                    # afa.get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)
                    get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)
                    source_error = None
                else:
                    files_rel = [str(Path(f).relative_to(ens.dir_source)) for f in source_files]
                    files_rel_lm =  list(landmarks_source)
                    if set(files_rel) != set(files_rel_lm):
                        print("Image files with landmarks do not match list of files in source folder matching requested pattern.")
                        print("Get landmarks using AFA.")
                        input_path = str(ens.dir_source) + os.sep
                        # afa.get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)                    
                        get_landmarks(input_path, file_prefix=pre, file_postfix=post, start_fresh=True)                    
                    source_error = None
    return source_error


def prepare_aligned(ens: Ensemble, ipd=None):
    pre=str(ens.file_bookends[0])
    post=str(ens.file_bookends[1])
    aligned_error = "Unknown"
    if ipd is None:
        if ens.INTER_PUPILLARY_DISTANCE is not None:
            ipd = ens.INTER_PUPILLARY_DISTANCE
        else:
            aligned_error = "INTER_PUPILLARY_DISTANCE needs to be specified when performing alignment."    
    # source_files = afa.make_files.get_source_files(ens.dir_source, pre, post)
    source_files = get_source_files(ens.dir_source, pre, post)
    files_rel = [str(Path(f).relative_to(ens.dir_source)) for f in source_files]
    if not ens.dir_aligned.exists():
        print("Alignment folder does not exist. Run alignment")
        if ipd is not None:
            adjust_size, size_value = "set_eye_distance", ipd
            input_path = str(ens.dir_source) + os.sep
            # aligned_path = afa.align_procrustes(input_path, adjust_size=adjust_size, size_value=size_value)
            aligned_path = align_procrustes(input_path, file_prefix=pre, file_postfix=post, adjust_size=adjust_size, size_value=size_value)
            # afa.get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
            get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
            aligned_error = None
            if ens.make_windowed_faces:
                no_save = False
                apresults, aperture_path = place_aperture(aligned_path, file_prefix=pre, file_postfix=post, no_save=no_save)
                assert Path(aperture_path) == ens.dir_windowed                
            else:
                no_save = True
                apresults = place_aperture(aligned_path, file_prefix=pre, file_postfix=post, no_save=no_save)
            ens.aperture = apresults
            ens.WINDOW = True
    else:
        # aligned_files = afa.make_files.get_source_files(ens.dir_aligned, pre, post)
        aligned_files = get_source_files(ens.dir_aligned, pre, post)
        if len(aligned_files) == 0:
            aligned_error = "No files in alignment folder matching requested pattern."
        else:
            files_rel_aligned = [str(Path(f).relative_to(ens.dir_aligned)) for f in aligned_files]
            if set(files_rel) != set(files_rel_aligned):  # files_rel from above "source" code
                aligned_error = "Mismatch between files in aligned and in source folders."
            else:
                specs_file = ens.dir_aligned / "specs.csv"
                if not specs_file.exists():
                    print("Alignment specification file (specs.csv) does not exist.")
                    print("Re-run alignment.")
                    if ipd is not None:
                        adjust_size, size_value = "set_eye_distance", ipd
                        input_path = str(ens.dir_source) + os.sep
                        # aligned_path = afa.align_procrustes(input_path, adjust_size=adjust_size, size_value=size_value)
                        aligned_path = align_procrustes(input_path, file_prefix=pre, file_postfix=post, adjust_size=adjust_size, size_value=size_value)
                        # afa.get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
                        get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
                        aligned_error = None
                        if ens.make_windowed_faces:
                            no_save = False
                            apresults, aperture_path = place_aperture(aligned_path, file_prefix=pre, file_postfix=post, no_save=no_save)
                            assert Path(aperture_path) == ens.dir_windowed                
                        else:
                            no_save = True
                            apresults = place_aperture(aligned_path, file_prefix=pre, file_postfix=post, no_save=no_save)
                        ens.aperture = apresults
                        ens.WINDOW = True
                else:
                    landmarks_file = ens.dir_aligned / "landmarks.txt"
                    if not landmarks_file.exists():
                        print("Get landmarks using AFA.")
                        aligned_path = str(ens.dir_aligned) + os.sep
                        # afa.get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
                        get_landmarks(aligned_path, file_prefix=pre, file_postfix=post)
                        aligned_error = None           
                    else:         
                        aligned_error = None
                        with open(ens.dir_aligned / "landmarks.txt") as json_data:
                            landmarks_aligned = json.load(json_data)
                            json_data.close()
                        ens.file_extension = list(landmarks_aligned)[0].split(".")[1]
                        if ens.file_extension != post:
                            aligned_error = "Aligned image files with landmarks have different file extension than specified."
                        files_rel_aligned_lm =  list(landmarks_aligned)
                        if set(files_rel_aligned) != set(files_rel_aligned_lm):
                            aligned_error = "Aligned image files with landmarks do not match list of files in aligned folder matching requested pattern."
                        if aligned_error == None:
                            aligned_path = str(ens.dir_aligned) + os.sep
                            if ens.make_windowed_faces:
                                no_save = False
                                apresults, aperture_path = place_aperture(aligned_path, file_prefix=pre, file_postfix=post, no_save=no_save)
                                assert Path(aperture_path) == ens.dir_windowed                
                            else:
                                no_save = True
                                apresults = place_aperture(aligned_path, file_prefix=pre, file_postfix=post, no_save=no_save)
                            ens.aperture = apresults
                            ens.WINDOW = True                                                        
    return aligned_error

def prepare_features(ens):
    pre=str(ens.file_bookends[0])
    post=str(ens.file_bookends[1])
    features_error = "Unknown"
    clone_directory_tree(ens.dir_aligned, new_dir='wrff', FilePrefix=pre, FilePostfix=post)
    aligned_files = get_source_files(ens.dir_aligned, pre, post)
    files_rel_aligned = [str(Path(f).relative_to(ens.dir_aligned)) for f in aligned_files]
    if not ens.dir_wrff.exists():
        print("WRFF folder does not exist. Creating.")
        ens.dir_wrff.mkdir(parents=False, exist_ok=False)        
        if ens.make_all_features:
            for f in list(ens.landmarks):
                make_features(ens, f)
        features_error = None
    else:
        # wrff_files = afa.make_files.get_source_files(ens.dir_wrff, pre, "png")
        wrff_files = get_source_files(ens.dir_wrff, pre, "png")
        if len(wrff_files) == 0:
            features_error = "No files in WRFF folder matching required pattern."
        else:
            files_rel_wrff = [str(Path(f).stem)[:-2] + "." + ens.file_extension for f in wrff_files]
            need_features = list(set(files_rel_aligned) - set(files_rel_wrff))  # files_rel_aligned from above "aligned" code
            files_rel_wrff = [str(Path(f).stem) for f in wrff_files]
            complete_files_wrff = [f.removesuffix(ens.file_extension)[:-1] + "-" + str(i+1) for i in range(ens.NUM_FEATURES) for f in files_rel_aligned]  # files_rel_aligned from above "aligned" code
            missing_a_feature = set(complete_files_wrff) - set(files_rel_wrff)
            missing_a_feature = set([n[:-2] for n in missing_a_feature])
            missing_a_feature = [f + "." + ens.file_extension for f in missing_a_feature]
            need_features.extend(missing_a_feature)
            need_features = set(need_features)
            for need in list(need_features):
                print("Features needed for: " + need)
                if ens.make_all_features:
                    make_features(ens, need)
            features_error = None
    return features_error




# Example usage
# CreativeCommons = Ensemble(dir_source = Path.cwd() / "faces", CriterionQuantileAmp=.5)
# CreativeCommons.list_faces()
# CreativeCommons.add_face("fabio-lanzoni")
# CreativeCommons.add_face("corey-haim")
# for person in CreativeCommons.Faces:
#     plt.imshow(person.img, cmap="gray")
#     plt.title(f'Original {person.name}')
#     plt.show()
#     plt.imshow(combine_features(person.F), cmap="gray")
#     plt.title(f'WRFF {person.name}')
#     plt.show()

# End 
# -------------------------------------------------------------------