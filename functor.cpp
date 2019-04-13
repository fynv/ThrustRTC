#include "functor.h"

std::string Functor::generate_code(const char* type_ret, const std::vector<const char*>& args) const
{
	std::string code;

	if (type_ret!=nullptr && functor_ret!=nullptr)
		code += std::string("    ") + type_ret + " " + functor_ret + ";\n";

	code+="    do{\n";

	for (size_t i = 0; i < functor_params.size(); i++)
		code += std::string("        auto ") + functor_params[i] + " = " + args[i] + "; \n";

	code += code_body;
	code += "    } while(false);\n";

	return code;

}
