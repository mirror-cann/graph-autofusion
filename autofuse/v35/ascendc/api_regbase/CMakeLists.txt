set(ascendc_api_regbase_extend_src
        concat.h
        datacopy.h
        broadcast.h
        datacopy_nddma.h
        gather.h
        clipbyvalue.h
        utils.h
        logical_not.h
        logical_xor.h
        cast.h
        logical.h
        compare.h
        floor_div.h
        pow.h
        split.h
        where.h
        where_v2.h
        erf.h
        erfinv.h
        exp2.h
        log1p.h
        tanh.h
        sub.h
        div.h
        sign.h
        reduce_init.h
        atan2.h
        airy_ai.h
        copy_sign.h
        erfcx.h
        expm.h
        fmod.h
        neg.h
        square.h
        trunc_div.h
        remainder.h
        bessel_j0.h
        bessel_j1.h
        modified_bessel_i0.h
        modified_bessel_i1.h
        modified_bessel_k0.h
        modified_bessel_k1.h
        modified_bessel_utils.h
        scaled_modified_bessel_k0.h
        scaled_modified_bessel_k1.h
        laguerre_polynomial_l.h
        legendre_polynomial_p.h
        igamma.h
        igammac.h
        trigonometric_function_utils.h
)

set(generated_regbase_files)

foreach(header ${ascendc_api_regbase_extend_src})
    string(REGEX REPLACE "\\.h$" "_reg_base.h" str_header ${header})
    add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMAND cat ${header}
            | sed "'1i\\R\"===('"
            | sed "'$$a\\)===\"'"
            > ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
            DEPENDS ${header}
    )
    list(APPEND generated_regbase_files ${CMAKE_CURRENT_BINARY_DIR}/${str_header})
endforeach()

add_custom_target(generated_regbase_files ALL DEPENDS ${generated_regbase_files})

add_library(ascendc_api_regbase_extend INTERFACE)
target_include_directories(ascendc_api_regbase_extend INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(ascendc_api_regbase_extend generated_regbase_files)