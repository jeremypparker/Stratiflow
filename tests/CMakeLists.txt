add_executable(tests FieldTests.cpp Tests.cpp DifferentiationTests.cpp IntegrationTests.cpp)
target_include_directories(tests PRIVATE ..)
target_link_libraries(tests StratiLib)