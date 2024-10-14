# wsi_processing

This code creates patches according to the conditions of parameters given from WSI (svs file) annotation (xml file).


### Environment
```
pip install -r requirements.txt
```
Above, I install python 3.6 with CUDA 11.4


### Anotation Detail
![image](https://user-images.githubusercontent.com/70703320/164376621-38718a6f-206a-4940-8cb5-4ea64dfb9888.png)

As shown in the figure above, the tumor exists only in the bounding box.


### xml File Example
```
<?xml version="1.0"?>
<ASAP_Annotations>
	<Annotations>
		<Annotation Name="vertical_boundary" Type="Rectangle" PartOfGroup="None" Color="#F4FA58">
			<Coordinates>
				<Coordinate Order="0" X="143" Y="113" />
				<Coordinate Order="1" X="300" Y="113" />
				<Coordinate Order="2" X="300" Y="346" />
				<Coordinate Order="3" X="143" Y="346" />
			</Coordinates>
		</Annotation>
		<Annotation Name="Annotation 1" Type="Spline" PartOfGroup="11_normal" Color="#F4FA58">
			<Coordinates>
				<Coordinate Order="0" X="151" Y="147" />
				<Coordinate Order="1" X="156" Y="150" />
				<Coordinate Order="2" X="162" Y="152" />
				<Coordinate Order="3" X="165" Y="153" />
				<Coordinate Order="4" X="168" Y="152" />
			</Coordinates>
		</Annotation>
		<Annotation Name="Annotation 2" Type="Spline" PartOfGroup="01_tumor1" Color="#F4FA58">
			<Coordinates>
				<Coordinate Order="0" X="263" Y="249" />
				<Coordinate Order="1" X="278" Y="249" />
				<Coordinate Order="2" X="278" Y="251" />
				<Coordinate Order="3" X="263" Y="251" />
			</Coordinates>
		</Annotation>
	</Annotations>
	<AnnotationGroups>
		<Group Name="11_normal" PartOfGroup="None" Color="#64FE2E">
			<Attributes />
		</Group>
		<Group Name="01_tumor1" PartOfGroup="None" Color="#64FE2E">
			<Attributes />
		</Group>
	</AnnotationGroups>
</ASAP_Annotations>
```

# Description

### Data Preparation
```
wsi_processing
        | # your prepration
        ├ WSI_anno_path
        |       ├ sample.svs
        |       └ sample.xml
        |
        | # our code
        ├ util_multi.py
        ├ get_patch_MP.py
        ├ requirements.txt
        └ README.md 
```

### Code Example
```
python get_patch_MP.py --slide_path './WSI_anno_path' --classes 4 --level 2 --anno_percent 0.05 --patch_size 512 --magnification 100 --save_patch_path './4classes_normal_path'
```

### Code Option
--`slide_path` : Input the target slide path. (ex. './WSI_anno_path') </br>
--`classes` : Input the class of the mask. (ex. 3 or 4) </br>
--`level` : Input the target level of the annotation mask to be referenced. (ex. 1 or 2) </br>
--`anno_percent` : Input the minimum percentage of annotations in a patch. (ex. 0.05) </br>
--`patch_size` : Input the image shape. (ex. 512 or 284 - for hooknet) </br>
--`magnification` : Input the target magnification of the patch. (ex. 50 or 100 or 200) </br>
--`save_patch_path` : Input the patch path that will be saved. (ex. './4classes_normal_path') </br>

# Output Example

### classes 3 & patch_size 284 (= HookNet)
![image](https://user-images.githubusercontent.com/70703320/164382582-3b0ad034-635c-4a30-849b-ddaa52391d71.png)

### classes 4 & patch_size 512
![image](https://user-images.githubusercontent.com/70703320/164383080-88f1be23-5730-4862-9077-903e118b1d71.png)