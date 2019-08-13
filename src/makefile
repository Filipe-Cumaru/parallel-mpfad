# Include the MOAB configuration information so that
# all required flags and libs are populated correctly
include makefile.config

default: all

ALLEXAMPLES = MPFADSolver
# TRILINOS_LIB_LINKS = -DMYAPP_EPETRA -L/home/facsa/Trilinos-SuperLU/lib -lamesos -laztecoo -ltrilinosss -ltriutils -lepetra -lepetraext -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lkokkoscore -lkokkoscore -lml
TRILINOS_LIB_LINKS = -DMYAPP_EPETRA -lamesos -laztecoo -ltrilinosss -ltriutils -lepetra -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lkokkoscore -lkokkoscore
# ${VERBOSE}$(MOAB_CXX) -o $@ main.cpp $< $(MOAB_LIBS_LINK) -std=c++11 -I$PWD -I/usr/include -I/home/facsa/Trilinos-SuperLU/include $(TRILINOS_LIB_LINKS)

MY_OBJS = geoutils.o LPEW3.o MPFADSolver.o

all: $(ALLEXAMPLES)

geoutils:
	@echo "[CXXLD]  $@"
	${VERBOSE}$(MOAB_CXX) -c -O3 geoutils.cpp $< $(MOAB_LIBS_LINK) -std=c++11 -I$PWD -I/usr/include

LPEW3:
	@echo "[CXXLD]  $@"
	${VERBOSE}$(MOAB_CXX) -c -O3 LPEW3.cpp $< $(MOAB_LIBS_LINK) -std=c++11 -I$PWD -I/usr/include

MPFADSolver:
	@echo "[CXXLD]  $@"
	${VERBOSE}$(MOAB_CXX) -c MPFADSolver.cpp $< $(MOAB_LIBS_LINK) -std=c++11 -I$PWD -I/usr/include $(TRILINOS_LIB_LINKS)

my_objs: geoutils LPEW3 MPFADSolver

main: main.o
	@echo "[CXXLD]  $@"
	${VERBOSE}$(MOAB_CXX) -o $@ $(MY_OBJS) $< $(MOAB_LIBS_LINK) -std=c++11 -I$PWD -I/usr/include $(TRILINOS_LIB_LINKS)

run: all $(addprefix run-,$(ALLEXAMPLES))

clean: clobber
	rm -rf ${ALLEXAMPLES}

teste:
	@echo "$@"
