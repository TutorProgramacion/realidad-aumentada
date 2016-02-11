INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src ${GNULIBS_INCLUDE_DIR})
LINK_LIBRARIES(${PROJECT_NAME} ${REQUIRED_LIBRARIES} )

ADD_EXECUTABLE(aruco_hrm_create_dictionary aruco_hrm_create_dictionary.cpp)
ADD_EXECUTABLE(aruco_hrm_test aruco_hrm_test.cpp)
ADD_EXECUTABLE(aruco_hrm_create_board aruco_hrm_create_board.cpp)
ADD_EXECUTABLE(aruco_hrm_test_board aruco_hrm_test_board.cpp)

set(ARUTILS_HRM aruco_hrm_create_dictionary aruco_hrm_test aruco_hrm_create_board aruco_hrm_test_board)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_target_properties(${ARUTILS_HRM} PROPERTIES FOLDER "ArUco/UtilsHrm")

INSTALL(TARGETS aruco_hrm_create_dictionary aruco_hrm_test aruco_hrm_create_board aruco_hrm_test_board RUNTIME DESTINATION bin)

