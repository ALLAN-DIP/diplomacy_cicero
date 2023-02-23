A **workset ID** for Diplomacy should be of format (in regex): dip[-\_a-zA-Z0-9]+[a-zA-Z0-9]<br>
The **workset filename** should be the same as the workset ID plus the .txt suffix, e.g. dip24.txt<br>
A workset file has two meta lines at the top. The meta lines start with a hashtag and a space ("# ").<br>
The first meta line has attributes such as ::type (always workset) ::id, ::username, ::date<br>
The second meta line contains a brief description of the workset (as value of attribute ::description).

A **sentence ID** should be of format (in regex): dip[\_a-zA-Z0-9]+[a-zA-Z0-9]\_\d\d\d\d\.\d+<br>

A **workset info filename** contains meta information such as sender, recipient, date<br>
The name of the _workset info filename_ is the same as the corresponding workset filename except with **_.info_** suffix instead of _.txt_ suffix.
Sentence IDs must match those in workset filename.

Example of workset dip24.txt<br>
```
# ::type workset ::id dip24 ::username dpeskov ::date Thu Mar 3, 2022
# ::description A short description about the workset
dip_0024.1 I propose an alliance against Turkey.
dip_0024.2 What do you say?
...
```

Example of workset dip24.info:
```
dip_0024.1 sender: turkey recipient: italy time: Spring 1901
dip_0024.2 sender: italy recipient: turkey time: Spring 1901
...
```
